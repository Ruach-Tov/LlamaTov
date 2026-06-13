#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""extract_pytorch_ir.py — Extract LLVM IR from PyTorch Inductor's generated C++.

For each op: torch.compile(op) → Inductor C++ → clang -emit-llvm → .ll file.
These .ll files are the PyTorch MKL reference IR for scan_ir_match.py.

Usage:
    python3 extract_pytorch_ir.py --output-dir /path/to/output/

Output: one .ll file per op in the output directory.
"""
import torch
import torch.nn.functional as F
import os
import sys
import glob
import subprocess
import shutil
import json

CLANG = "/nix/store/4kb26qqjpf23gmi26ddbp86g0cfj3l6p-clang-wrapper-19.1.7/bin/clang"

# Ops to extract — each is a (name, function, input_generator)
OPS = [
    ("relu",        lambda x: torch.relu(x),           lambda: torch.randn(1024)),
    ("silu",        lambda x: torch.nn.functional.silu(x), lambda: torch.randn(1024)),
    ("sigmoid",     lambda x: torch.sigmoid(x),        lambda: torch.randn(1024)),
    ("tanh",        lambda x: torch.tanh(x),           lambda: torch.randn(1024)),
    ("gelu_exact",  lambda x: F.gelu(x),               lambda: torch.randn(1024)),
    ("gelu_tanh",   lambda x: F.gelu(x, approximate='tanh'), lambda: torch.randn(1024)),
    ("softplus",    lambda x: F.softplus(x),           lambda: torch.randn(1024)),
    ("leaky_relu",  lambda x: F.leaky_relu(x, 0.01),  lambda: torch.randn(1024)),
    ("elu",         lambda x: F.elu(x),                lambda: torch.randn(1024)),
    ("softsign",    lambda x: F.softsign(x),           lambda: torch.randn(1024)),
    ("hardsigmoid", lambda x: F.hardsigmoid(x),       lambda: torch.randn(1024)),
    ("softmax",     lambda x: F.softmax(x, dim=0),     lambda: torch.randn(1024)),
    ("log_softmax", lambda x: F.log_softmax(x, dim=0), lambda: torch.randn(1024)),
    ("scale",       lambda x: x * 0.5,                 lambda: torch.randn(1024)),
    ("cumsum",      lambda x: torch.cumsum(x, dim=0),  lambda: torch.randn(1024)),
]


def extract_inductor_cpp(name, fn, input_gen):
    """Compile an op with Inductor and return the generated C++ path."""
    try:
        # Snapshot existing .cpp files in torchinductor dirs
        user = os.environ.get("USER", "www")
        search_dirs = [f"/tmp/torchinductor_{user}/", "/tmp/torchinductor_www/"]
        
        existing = set()
        for d in search_dirs:
            if os.path.isdir(d):
                existing.update(glob.glob(os.path.join(d, "**/*.cpp"), recursive=True))
        
        x = input_gen()
        compiled = torch.compile(fn, backend="inductor")
        result = compiled(x)
        
        # Find NEW .cpp files
        for d in search_dirs:
            if os.path.isdir(d):
                all_cpp = glob.glob(os.path.join(d, "**/*.cpp"), recursive=True)
                new_files = [f for f in all_cpp if f not in existing]
                if new_files:
                    new_files.sort(key=os.path.getmtime, reverse=True)
                    return new_files[0]
        
        # If no new files, return the most recent one
        for d in search_dirs:
            if os.path.isdir(d):
                all_cpp = glob.glob(os.path.join(d, "**/*.cpp"), recursive=True)
                if all_cpp:
                    all_cpp.sort(key=os.path.getmtime, reverse=True)
                    return all_cpp[0]
    except Exception as e:
        print(f"  {name}: inductor failed: {e}", flush=True)
    
    return None


def cpp_to_llvm_ir(cpp_path, ll_path):
    """Compile Inductor C++ to LLVM IR."""
    # Extract just the kernel function — strip Python bindings
    with open(cpp_path) as f:
        code = f.read()
    
    # Find the kernel function
    kernel_start = code.find('extern "C"')
    if kernel_start < 0:
        kernel_start = code.find('void kernel(')
    if kernel_start < 0:
        return False
    
    # Find the end of the kernel (matching braces)
    brace_count = 0
    kernel_end = kernel_start
    found_first = False
    for i in range(kernel_start, len(code)):
        if code[i] == '{':
            brace_count += 1
            found_first = True
        elif code[i] == '}':
            brace_count -= 1
            if found_first and brace_count == 0:
                kernel_end = i + 1
                break
    
    kernel_code = code[kernel_start:kernel_end]
    
    # Get the header include
    header_line = ""
    for line in code.split('\n'):
        if '#include' in line and 'torchinductor' in line:
            # Read the header content inline
            header_path = line.split('"')[1]
            if os.path.exists(header_path):
                with open(header_path) as hf:
                    header_line = hf.read()
            break
    
    # Convert C++ to pure C — replace std::max/min/exp/etc with C equivalents
    kernel_code = kernel_code.replace('std::max', 'fmaxf')
    kernel_code = kernel_code.replace('std::min', 'fminf')
    kernel_code = kernel_code.replace('std::exp', 'expf')
    kernel_code = kernel_code.replace('std::log', 'logf')
    kernel_code = kernel_code.replace('std::tanh', 'tanhf')
    kernel_code = kernel_code.replace('std::abs', 'fabsf')
    kernel_code = kernel_code.replace('std::sqrt', 'sqrtf')
    kernel_code = kernel_code.replace('std::erf', 'erff')
    kernel_code = kernel_code.replace('static_cast<int64_t>', '(long)')
    kernel_code = kernel_code.replace('decltype(tmp0)(0)', '0.0f')
    kernel_code = kernel_code.replace('decltype(tmp0)(1)', '1.0f')
    kernel_code = kernel_code.replace('int64_t', 'long')
    # Remove #pragma GCC ivdep
    kernel_code = '\n'.join(l for l in kernel_code.split('\n') if '#pragma' not in l)

    # Write a standalone C file (not C++)
    standalone = f"""
#include <math.h>

{kernel_code}
"""
    
    standalone_path = ll_path.replace('.ll', '.c')
    with open(standalone_path, 'w') as f:
        f.write(standalone)
    
    # Compile to LLVM IR
    r = subprocess.run(
        [CLANG, "-S", "-emit-llvm", "-O2",
         "-msse3", "-mno-avx", "-mno-fma",
         "-o", ll_path, standalone_path],
        capture_output=True, text=True, timeout=10
    )
    
    if r.returncode != 0:
        print(f"    clang error: {r.stderr[:200]}", flush=True)
    
    return r.returncode == 0


def main():
    output_dir = "/tmp/bpd-generated/build/pytorch_ir"
    if "--output-dir" in sys.argv:
        output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("Extracting PyTorch Inductor IR...", flush=True)
    for name, fn, input_gen in OPS:
        cpp_path = extract_inductor_cpp(name, fn, input_gen)
        if not cpp_path:
            print(f"  {name}: no C++ generated", flush=True)
            continue
        
        ll_path = os.path.join(output_dir, f"pytorch_{name}.ll")
        ok = cpp_to_llvm_ir(cpp_path, ll_path)
        
        if ok and os.path.exists(ll_path):
            size = os.path.getsize(ll_path)
            print(f"  {name}: {ll_path} ({size} bytes)", flush=True)
            results[name] = ll_path
        else:
            print(f"  {name}: clang compile failed", flush=True)
    
    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{len(results)}/{len(OPS)} ops extracted. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
