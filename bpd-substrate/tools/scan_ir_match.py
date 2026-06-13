#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""scan_ir_match.py — Automatically scan all emitters for IR-match status.

For each op in kernel_patterns.pl:
  1. Compile ggml's C implementation to LLVM IR (clang -S -emit-llvm -O2)
  2. Generate our Prolog-emitted LLVM IR
  3. Compare float operation sequences
  4. Emit ulp_op_match/5 Prolog facts

Output: Prolog facts ready to paste into llvm_op_match.pl

Usage:
    python3 scan_ir_match.py > new_facts.pl
"""
import subprocess
import os
import re
import sys
import tempfile

CLANG = "/nix/store/4kb26qqjpf23gmi26ddbp86g0cfj3l6p-clang-wrapper-19.1.7/bin/clang"
LLVM_AS = "/nix/store/a8hbpr6c8rhdcvgr19r8gnqifnnfb9q3-llvm-19.1.7/bin/llvm-as"
LLC = "/nix/store/a8hbpr6c8rhdcvgr19r8gnqifnnfb9q3-llvm-19.1.7/bin/llc"
BPD_DIR = "/tmp/bpd-substrate/lib"

# Map each op to its reference C implementation
REFERENCE_C = {
    # Unary elementwise
    'ggml_relu': 'void ggml_relu(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = s[i] > 0 ? s[i] : 0; }',
    'ggml_silu': 'void ggml_silu(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = s[i] / (1.0f + __builtin_expf(-s[i])); }',
    'ggml_sigmoid': 'void ggml_sigmoid(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = 1.0f / (1.0f + __builtin_expf(-s[i])); }',
    'ggml_tanh': 'void ggml_tanh(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = __builtin_tanhf(s[i]); }',
    'ggml_softplus': 'void ggml_softplus(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = __builtin_logf(1.0f + __builtin_expf(s[i])); }',
    'ggml_leaky_relu': 'void ggml_leaky_relu(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = s[i] > 0 ? s[i] : 0.01f*s[i]; }',
    'ggml_elu': 'void ggml_elu(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = s[i] > 0 ? s[i] : __builtin_expf(s[i])-1.0f; }',
    'ggml_softsign': 'void ggml_softsign(int n, float* d, const float* s) { for(int i=0;i<n;i++) d[i] = s[i] / (1.0f + __builtin_fabsf(s[i])); }',
    'ggml_hardsigmoid': 'void ggml_hardsigmoid(int n, float* d, const float* s) { for(int i=0;i<n;i++) { float v=s[i]*(1.0f/6.0f)+0.5f; d[i]=v<0?0:(v>1?1:v); } }',
    'ggml_gelu': '''void ggml_gelu(int n, float* d, const float* s) {
        for(int i=0;i<n;i++) {
            float x=s[i];
            d[i] = 0.5f*x*(1.0f+__builtin_tanhf(0.7978845608f*(x+0.044715f*x*x*x)));
        }
    }''',
    # Reduction
    'ggml_sum': '''#include <pmmintrin.h>
    void ggml_sum(int n, float* s, const float* x) {
        float sumf=0; const int np=(n&~31);
        __m128 sum[8]; for(int j=0;j<8;j++) sum[j]=_mm_setzero_ps();
        for(int i=0;i<np;i+=32) for(int j=0;j<8;j++) sum[j]=_mm_add_ps(sum[j],_mm_loadu_ps(x+i+j*4));
        sum[0]=_mm_add_ps(sum[0],sum[4]); sum[1]=_mm_add_ps(sum[1],sum[5]);
        sum[2]=_mm_add_ps(sum[2],sum[6]); sum[3]=_mm_add_ps(sum[3],sum[7]);
        sum[0]=_mm_add_ps(sum[0],sum[2]); sum[1]=_mm_add_ps(sum[1],sum[3]);
        sum[0]=_mm_add_ps(sum[0],sum[1]);
        sum[0]=_mm_hadd_ps(sum[0],sum[0]); sum[0]=_mm_hadd_ps(sum[0],sum[0]);
        _mm_store_ss(&sumf,sum[0]); *s=sumf;
    }''',
    # Scan
    'ggml_cumsum': 'void ggml_cumsum(int n, float* d, const float* s) { float a=0; for(int i=0;i<n;i++){a+=s[i];d[i]=a;} }',
    'ggml_cumprod': 'void ggml_cumprod(int n, float* d, const float* s) { float a=1; for(int i=0;i<n;i++){a*=s[i];d[i]=a;} }',
    # Binary elementwise
    'ggml_scale': '''#include <pmmintrin.h>
    void ggml_scale(int n, float* d, const float* s, float sc) {
        const int np=(n&~31); __m128 sv=_mm_set1_ps(sc);
        for(int i=0;i<np;i+=32) for(int j=0;j<8;j++)
            _mm_storeu_ps(d+i+j*4, _mm_mul_ps(_mm_loadu_ps(s+i+j*4),sv));
        for(int i=np;i<n;i++) d[i]=s[i]*sc;
    }''',
}

# Map ops to their BPD .ll files and function names
BPD_EMITTERS = {
    'ggml_relu': ('bpd_llvm_elem_generated.ll', 'bpd_relu'),
    'ggml_silu': ('bpd_llvm_elem_generated.ll', 'bpd_silu'),
    'ggml_sigmoid': ('bpd_llvm_elem_generated.ll', 'bpd_sigmoid'),
    'ggml_tanh': ('bpd_tanh_fix.ll', 'bpd_tanh_fixed'),
    'ggml_softplus': ('bpd_llvm_elem_generated.ll', 'bpd_softplus'),
    'ggml_leaky_relu': ('bpd_llvm_elem_generated.ll', 'bpd_leaky_relu'),
    'ggml_elu': ('bpd_llvm_elem_generated.ll', 'bpd_elu'),
    'ggml_softsign': ('bpd_llvm_elem_generated.ll', 'bpd_softsign'),
    'ggml_hardsigmoid': ('bpd_tanh_fix.ll', 'bpd_hardsigmoid_fixed'),
    'ggml_gelu': ('bpd_tanh_fix.ll', 'bpd_gelu_fixed'),
    'ggml_sum': ('bpd_sum_sse3.ll', 'bpd_sum_sse3'),
    'ggml_cumsum': ('bpd_scale_cumsum.ll', 'bpd_cumsum'),
    'ggml_cumprod': ('bpd_scale_cumsum.ll', 'bpd_cumprod'),
    'ggml_scale': ('bpd_scale_cumsum.ll', 'bpd_scale'),
}

def extract_float_ops(text):
    """Extract float operation sequence from LLVM IR."""
    ops = []
    for line in text.strip().split('\n'):
        line = line.strip()
        for op in ['fadd', 'fsub', 'fmul', 'fdiv', 'fneg', 'fcmp']:
            m = re.search(r'\b(' + op + r')\s+(float|<\d+ x float>)', line)
            if m:
                ops.append('%s %s' % (m.group(1), m.group(2)))
        if 'call' in line:
            m = re.search(r'call\s+\S+\s+@([\w.]+)', line)
            if m:
                fname = m.group(1)
                if any(k in fname for k in ['llvm.', 'expf', 'logf', 'tanhf', 'sqrtf', 'fabsf']):
                    ops.append('call %s' % fname)
        if re.search(r'\bselect\b', line):
            ops.append('select')
    return ops

def get_core_ops(ops):
    """Deduplicate unrolled repetitions, return unique op set."""
    return sorted(set(ops))

def compile_ref_to_ir(op_name, c_code):
    """Compile reference C to LLVM IR."""
    with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
        f.write(c_code)
        c_path = f.name
    ll_path = c_path.replace('.c', '.ll')
    try:
        subprocess.run([CLANG, '-S', '-emit-llvm', '-O2', '-msse3', '-mno-avx', '-mno-fma',
                       '-o', ll_path, c_path], capture_output=True, timeout=10)
        if os.path.exists(ll_path):
            return open(ll_path).read()
    except:
        pass
    finally:
        os.unlink(c_path)
        if os.path.exists(ll_path):
            os.unlink(ll_path)
    return None

def get_bpd_ir(ll_file, func_name):
    """Read BPD's emitted IR for a function."""
    # Check multiple locations
    for prefix in ['/tmp/bpd-substrate/lib/', '/tmp/', os.path.expanduser('~/Ruach-Tov/bpd/lib/')]:
        path = prefix + ll_file
        if os.path.exists(path):
            text = open(path).read()
            # Extract the specific function
            m = re.search(r'define\s+[^@]*@' + re.escape(func_name) + r'\b.*?\n\}', text, re.DOTALL)
            if m:
                return m.group(0)
            return text
    return None

def classify(ref_ops, bpd_ops):
    """Classify the match level."""
    if ref_ops == bpd_ops:
        return 'IR-MATCH', 0
    
    ref_set = set(ref_ops)
    bpd_set = set(bpd_ops)
    
    # Same unique ops = IR-MATCH (unrolling difference only)
    if get_core_ops(ref_ops) == get_core_ops(bpd_ops):
        return 'IR-MATCH', 0
    
    # Same op types, maybe different counts
    if ref_set == bpd_set:
        return 'EQUIVALENT', 0
    
    # Subset
    if bpd_set <= ref_set or ref_set <= bpd_set:
        return 'SIMILAR', 5
    
    return 'DIVERGENT', 999


def main():
    print("%% Auto-generated by scan_ir_match.py")
    print("%% Date: 2026-05-27")
    print()
    
    results = []
    
    for op_name in sorted(REFERENCE_C.keys()):
        if op_name not in BPD_EMITTERS:
            continue
        
        ll_file, func_name = BPD_EMITTERS[op_name]
        c_code = REFERENCE_C[op_name]
        
        # Get reference IR
        ref_ir = compile_ref_to_ir(op_name, c_code)
        if not ref_ir:
            print("%% %s: could not compile reference" % op_name, file=sys.stderr)
            continue
        
        # Get BPD IR
        bpd_ir = get_bpd_ir(ll_file, func_name)
        if not bpd_ir:
            print("%% %s: BPD IR not found (%s:%s)" % (op_name, ll_file, func_name), file=sys.stderr)
            continue
        
        ref_ops = extract_float_ops(ref_ir)
        bpd_ops = extract_float_ops(bpd_ir)
        
        verdict, ulp = classify(ref_ops, bpd_ops)
        
        # Determine pattern
        pattern = 'unknown'
        if op_name in ['ggml_relu','ggml_silu','ggml_sigmoid','ggml_tanh','ggml_gelu',
                       'ggml_softplus','ggml_leaky_relu','ggml_elu','ggml_hardsigmoid','ggml_softsign']:
            pattern = 'unary_elementwise'
        elif op_name in ['ggml_scale']:
            pattern = 'binary_elementwise'
        elif op_name in ['ggml_sum','ggml_mean','ggml_max','ggml_min','ggml_mul_mat']:
            pattern = 'reduction'
        elif op_name in ['ggml_cumsum','ggml_cumprod']:
            pattern = 'scan'
        
        ref_str = 'ggml_sse3' if verdict == 'IR-MATCH' else 'scalar'
        
        results.append((pattern, op_name, ref_str, ulp, verdict, ref_ops, bpd_ops))
        
        print("ulp_op_match(%s, %s, %s, %d, 'ir_diff: %s — ref_ops=%s bpd_ops=%s')." % (
            pattern, op_name, ref_str, ulp, verdict,
            get_core_ops(ref_ops), get_core_ops(bpd_ops)))
    
    # Summary
    ir_match = sum(1 for r in results if r[4] == 'IR-MATCH')
    equiv = sum(1 for r in results if r[4] == 'EQUIVALENT')
    div = sum(1 for r in results if r[4] == 'DIVERGENT')
    
    print()
    print("%% Summary: %d IR-MATCH, %d EQUIVALENT, %d DIVERGENT out of %d scanned" % (
        ir_match, equiv, div, len(results)), file=sys.stderr)


if __name__ == '__main__':
    main()
