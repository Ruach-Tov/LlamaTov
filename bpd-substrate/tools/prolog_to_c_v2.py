#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""prolog_to_c_v2.py — Generate C from Prolog graph + routing table.

Uses the layer-level routing (route/2 facts) from YOLOv5's forward pass
to emit ops in the correct order with skip connections properly wired.
"""
import sys, re, struct
from collections import defaultdict, OrderedDict


GGML_DISPATCH = {
    'relu': 'ggml_relu', 'leaky_relu': 'ggml_leaky_relu',
    'silu': 'ggml_silu', 'gelu': 'ggml_gelu',
    'sigmoid': 'ggml_sigmoid', 'tanh': 'ggml_tanh',
    'elu': 'ggml_elu', 'hardswish': 'ggml_hardswish',
    'hardsigmoid': 'ggml_hardsigmoid',
    'abs': 'ggml_abs', 'neg': 'ggml_neg', 'sqrt': 'ggml_sqrt',
}


def parse_prolog(filename):
    ops = OrderedDict(); outputs = {}; inputs = {}; attrs = defaultdict(dict)
    routes = {}; saves = set(); detect_inputs = []
    
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%') or line.startswith(':-'): continue
            
            m = re.match(r"op_kind\((\w+),\s*(\w+)\)\.", line)
            if m: ops[m.group(1)] = m.group(2); continue
            m = re.match(r"op_output\((\w+),\s*(\w+)\)\.", line)
            if m: outputs[m.group(1)] = m.group(2); continue
            m = re.match(r"op_inputs\((\w+),\s*\[([^\]]*)\]\)\.", line)
            if m: inputs[m.group(1)] = [x.strip() for x in m.group(2).split(',') if x.strip()]; continue
            m = re.match(r"op_attr\((\w+),\s*(\w+),\s*(.+)\)\.", line)
            if m:
                val = m.group(3)
                try: val = int(val)
                except:
                    try: val = float(val)
                    except: val = val.strip("'\"")
                attrs[m.group(1)][m.group(2)] = val
                continue
            m = re.match(r"route\((\d+),\s*\[([^\]]*)\]\)", line)
            if m: routes[int(m.group(1))] = [int(x.strip()) for x in m.group(2).split(',')]; continue
            m = re.match(r"save_layer\((\d+)\)", line)
            if m: saves.add(int(m.group(1))); continue
            if line.startswith("detect_inputs"):
                nums = re.findall(r"\d+", line)
                detect_inputs = [int(x) for x in nums]; continue
    
    return ops, outputs, inputs, attrs, routes, saves, detect_inputs


def get_layer_ops(ops, layer_idx):
    """Get all op names belonging to a layer (model_N_*)."""
    prefix = "model_%d_" % layer_idx
    exact = "model_%d" % layer_idx
    result = []
    for name in ops:
        if name == exact or name.startswith(prefix):
            result.append(name)
    return result


def emit_layer_ops(c, ops, attrs, layer_ops, layer_idx, inp_var, model_prefix="m"):
    """Emit ggml calls for all ops in a layer, return output variable name."""
    out_var = "layer_%d" % layer_idx
    cur = inp_var
    
    for name in layer_ops:
        kind = ops[name]
        a = attrs.get(name, {})
        
        if kind == 'conv2d':
            s = a.get('stride', 1); p = a.get('padding', 0)
            c.append("    %s = ggml_conv_2d(ctx, %s->%s_weight, %s, %d, %d, %d, %d, 1, 1);" % (
                out_var, model_prefix, name, cur, s, s, p, p))
            cur = out_var
        elif kind == 'batchnorm':
            eps = a.get('eps', 0.001)
            c.append("    %s = ggml_norm(ctx, %s, %sf);" % (out_var, cur, eps))
            c.append("    %s = ggml_mul(ctx, %s, %s->%s_weight);" % (out_var, out_var, model_prefix, name))
            c.append("    %s = ggml_add(ctx, %s, %s->%s_bias);" % (out_var, out_var, model_prefix, name))
            cur = out_var
        elif kind in GGML_DISPATCH:
            c.append("    %s = %s(ctx, %s);" % (out_var, GGML_DISPATCH[kind], cur))
            cur = out_var
        elif kind == 'add':
            # Residual add — second input is the layer's own input (skip connection)
            c.append("    %s = ggml_add(ctx, %s, %s);  /* residual */" % (out_var, cur, inp_var))
            cur = out_var
        elif kind == 'maxpool':
            ks = a.get('kernel_size', 5); st = a.get('stride', 1); pd = ks // 2
            c.append("    %s = ggml_pool_2d(ctx, %s, GGML_OP_POOL_MAX, %d, %d, %d, %d, %d, %d);" % (
                out_var, cur, ks, ks, st, st, pd, pd))
            cur = out_var
        elif kind == 'concat':
            pass  # Handled at the layer level
        elif kind == 'upsample':
            pass  # Handled at the layer level
        elif kind == 'detect':
            pass  # Handled at the layer level
        elif kind == 'cbs':
            # Compound: walk sub-ops
            pass
        elif kind == 'c3':
            pass
        elif kind == 'sppf':
            pass
    
    return out_var



def _emit_single_op(c, ops, attrs, op_name, cur_var, inp_var, model_prefix):
    """Emit a single ggml call for one op. Returns the new cur variable name."""
    kind = ops[op_name]
    a = attrs.get(op_name, {})
    
    if kind == 'conv2d':
        s = a.get('stride', 1); p = a.get('padding', 0)
        c.append('    cur = ggml_conv_2d(ctx, %s->%s_weight, %s, %d, %d, %d, %d, 1, 1);' % (
            model_prefix, op_name, cur_var, s, s, p, p))
        return 'cur'
    elif kind == 'batchnorm':
        eps = a.get('eps', 0.001)
        c.append('    cur = ggml_norm(ctx, %s, %sf);' % (cur_var, eps))
        c.append('    cur = ggml_mul(ctx, cur, %s->%s_weight);' % (model_prefix, op_name))
        c.append('    cur = ggml_add(ctx, cur, %s->%s_bias);' % (model_prefix, op_name))
        return 'cur'
    elif kind in GGML_DISPATCH:
        c.append('    cur = %s(ctx, %s);' % (GGML_DISPATCH[kind], cur_var))
        return 'cur'
    elif kind == 'add':
        c.append('    cur = ggml_add(ctx, cur, %s);  /* residual */' % inp_var)
        return 'cur'
    return cur_var

def emit_c(ops, outputs, inputs, attrs, routes, saves, detect_inputs, output_file):
    c = []
    c.append('/* Auto-generated by prolog_to_c_v2.py from Prolog graph facts.')
    c.append(' * Uses layer-level routing for correct skip connection wiring.')
    c.append(' * BPD Substrate: https://github.com/heath-hunnicutt-ruach-tov/bpd-substrate')
    c.append(' */')
    c.append('#include "ggml.h"')
    c.append('#include <stdio.h>')
    c.append('#include <stdlib.h>')
    c.append('#include <string.h>')
    c.append('')
    
    # Model struct — weight tensors
    c.append('struct model {')
    c.append('    struct ggml_context * ctx_w;')
    
    for name, kind in ops.items():
        a = attrs.get(name, {})
        if kind == 'conv2d':
            ic = a.get('in_channels', 0); oc = a.get('out_channels', 0); ks = a.get('kernel_size', 1)
            c.append('    struct ggml_tensor * %s_weight;  /* [%d,%d,%d,%d] */' % (name, oc, ic, ks, ks))
        elif kind == 'batchnorm':
            nf = a.get('num_features', 0)
            c.append('    struct ggml_tensor * %s_weight;  /* [%d] */' % (name, nf))
            c.append('    struct ggml_tensor * %s_bias;' % name)
            c.append('    struct ggml_tensor * %s_mean;' % name)
            c.append('    struct ggml_tensor * %s_var;' % name)
    c.append('};')
    c.append('')
    
    # model_init
    c.append('void model_init(struct model * m) {')
    c.append('    struct ggml_init_params p = { .mem_size = 256*1024*1024, .mem_buffer = NULL, .no_alloc = false };')
    c.append('    m->ctx_w = ggml_init(p);')
    for name, kind in ops.items():
        a = attrs.get(name, {})
        if kind == 'conv2d':
            ic = a.get('in_channels', 3); oc = a.get('out_channels', 16); ks = a.get('kernel_size', 3)
            c.append('    m->%s_weight = ggml_new_tensor_4d(m->ctx_w, GGML_TYPE_F32, %d, %d, %d, %d);' % (name, ks, ks, ic, oc))
            c.append('    ggml_set_name(m->%s_weight, "%s_weight");' % (name, name))
        elif kind == 'batchnorm':
            nf = a.get('num_features', 16)
            for suffix in ['weight', 'bias', 'mean', 'var']:
                c.append('    m->%s_%s = ggml_new_tensor_1d(m->ctx_w, GGML_TYPE_F32, %d);' % (name, suffix, nf))
    c.append('}')
    c.append('')
    
    # Forward pass using LAYER-LEVEL routing
    c.append('struct ggml_tensor * model_forward(struct model * m, struct ggml_context * ctx, struct ggml_tensor * input) {')
    n_layers = max(routes.keys()) + 1 if routes else 25
    c.append('    struct ggml_tensor * layer[%d];' % n_layers)
    c.append('    struct ggml_tensor * cur;')
    c.append('')
    
    for i in range(n_layers):
        layer_ops = get_layer_ops(ops, i)
        if not layer_ops:
            continue
        
        route = routes.get(i, [-1])
        layer_type = None
        for op in layer_ops:
            if ops[op] in ('cbs', 'c3', 'sppf', 'concat', 'upsample', 'detect'):
                layer_type = ops[op]
                break
        if layer_type is None:
            # Check if it's a simple conv layer
            for op in layer_ops:
                if ops[op] == 'conv2d':
                    layer_type = 'cbs'  # Conv layers are typically CBS
                    break
        
        # Determine input
        if route == [-1]:
            if i == 0:
                inp = 'input'
            else:
                inp = 'layer[%d]' % (i - 1)
        elif len(route) == 1 and route[0] >= 0:
            inp = 'layer[%d]' % route[0]
        else:
            # Multi-input (concat or detect)
            inp = None  # Handled specially
        
        c.append('    /* Layer %d: %s */' % (i, layer_type or 'unknown'))
        
        if layer_type == 'concat' and inp is None:
            # Concat from multiple sources
            srcs = []
            for r in route:
                if r == -1:
                    srcs.append('layer[%d]' % (i - 1))
                else:
                    srcs.append('layer[%d]' % r)
            c.append('    layer[%d] = ggml_concat(ctx, %s, %s, 2);' % (i, srcs[0], srcs[1]))
            
        elif layer_type == 'upsample':
            c.append('    layer[%d] = ggml_upscale(ctx, %s, 2, GGML_SCALE_MODE_NEAREST);' % (i, inp))
            
        elif layer_type == 'detect' and detect_inputs:
            c.append('    /* Detect heads — outputs from layers %s */' % detect_inputs)
            for j, det_layer in enumerate(detect_inputs):
                head_ops = [op for op in layer_ops if 'head%d' % j in op and ops[op] == 'conv2d']
                if head_ops:
                    a = attrs.get(head_ops[0], {})
                    s = a.get('stride', 1); p = a.get('padding', 0)
                    c.append('    /* detect head %d from layer %d */' % (j, det_layer))
                    c.append('    layer[%d] = ggml_conv_2d(ctx, m->%s_weight, layer[%d], %d, %d, %d, %d, 1, 1);' % (
                        i, head_ops[0], det_layer, s, s, p, p))
                    
        elif layer_type == 'c3':
            # C3: cv1(input) → bottlenecks → concat(bn_out, cv2(input)) → cv3
            cv1_ops = [op for op in layer_ops if '_cv1_' in op]
            cv2_ops = [op for op in layer_ops if '_cv2_' in op and '_cv2_' in op.split('bot')[0] if 'bot' not in op else True]
            cv3_ops = [op for op in layer_ops if '_cv3_' in op]
            bot_ops = [op for op in layer_ops if '_bot' in op]
            
            # Separate cv2 (the skip branch) from cv2 inside bottlenecks
            cv2_branch = [op for op in layer_ops if op.endswith('_cv2_conv') or op.endswith('_cv2_bn') or op.endswith('_cv2_act') 
                         if 'bot' not in op]
            
            # cv1 branch
            cur = inp
            for op_name in sorted(cv1_ops, key=lambda x: ('conv' in x, 'bn' in x, 'act' in x)):
                cur = _emit_single_op(c, ops, attrs, op_name, cur, inp, 'm')
            cv1_out = 'cur'
            
            # Bottleneck chain
            bn_groups = {}
            for op in bot_ops:
                # Extract bottleneck index: model_2_bot0_cv1_conv → bot0
                m_bn = re.search(r'_bot(\d+)_', op)
                if m_bn:
                    bn_idx = int(m_bn.group(1))
                    bn_groups.setdefault(bn_idx, []).append(op)
            
            bn_input = cv1_out
            for bn_idx in sorted(bn_groups.keys()):
                bn_ops = sorted(bn_groups[bn_idx], key=lambda x: ('cv1' in x, 'cv2' in x, 'conv' in x, 'bn' in x))
                bn_start = bn_input  # save for residual
                for op_name in bn_ops:
                    kind = ops[op_name]
                    if kind == 'add':
                        c.append('    cur = ggml_add(ctx, cur, %s);  /* bottleneck residual */' % bn_start)
                    else:
                        cur = _emit_single_op(c, ops, attrs, op_name, 'cur', bn_start, 'm')
                bn_input = 'cur'
            bottleneck_out = 'cur'
            
            # cv2 branch (from ORIGINAL input, not from bottleneck output)
            c.append('    /* cv2 branch (from layer input) */')
            cv2_cur = inp
            for op_name in sorted(cv2_branch, key=lambda x: ('conv' in x, 'bn' in x, 'act' in x)):
                cv2_cur = _emit_single_op(c, ops, attrs, op_name, cv2_cur, inp, 'm')
            
            # Concat
            c.append('    cur = ggml_concat(ctx, %s, %s, 2);  /* C3 concat */' % (bottleneck_out, cv2_cur))
            
            # cv3
            for op_name in sorted(cv3_ops, key=lambda x: ('conv' in x, 'bn' in x, 'act' in x)):
                cur = _emit_single_op(c, ops, attrs, op_name, 'cur', inp, 'm')
            
            c.append('    layer[%d] = cur;' % i)
            
        elif layer_type == 'sppf':
            # SPPF: cv1(x) → pool → pool → pool → concat(y,p1,p2,p3) → cv2
            cv1_ops = [op for op in layer_ops if '_cv1_' in op]
            cv2_ops = [op for op in layer_ops if '_cv2_' in op]
            pool_ops = [op for op in layer_ops if ops.get(op) == 'maxpool']
            
            # cv1
            cur = inp
            for op_name in sorted(cv1_ops, key=lambda x: ('conv' in x, 'bn' in x, 'act' in x)):
                cur = _emit_single_op(c, ops, attrs, op_name, cur, inp, 'm')
            c.append('    { /* SPPF pool chain */')
            c.append('    struct ggml_tensor * y = cur;')
            
            # Pool chain
            pool_ops_sorted = sorted(pool_ops)
            for j, pname in enumerate(pool_ops_sorted):
                a = attrs.get(pname, {})
                ks = a.get('kernel_size', 5); pd = ks // 2
                c.append('    struct ggml_tensor * p%d = ggml_pool_2d(ctx, %s, GGML_OP_POOL_MAX, %d, %d, 1, 1, %d, %d);' % (
                    j+1, 'y' if j == 0 else 'p%d' % j, ks, ks, pd, pd))
            
            # Concat y + p1 + p2 + p3
            n_pools = len(pool_ops_sorted)
            c.append('    cur = ggml_concat(ctx, y, p1, 2);')
            for j in range(2, n_pools + 1):
                c.append('    cur = ggml_concat(ctx, cur, p%d, 2);' % j)
            c.append('    }')
            
            # cv2
            for op_name in sorted(cv2_ops, key=lambda x: ('conv' in x, 'bn' in x, 'act' in x)):
                cur = _emit_single_op(c, ops, attrs, op_name, 'cur', inp, 'm')
            
            c.append('    layer[%d] = cur;' % i)
            
        elif layer_type in ('cbs', None):
            # Simple sequential: walk sub-ops in order
            cur = inp
            for op_name in layer_ops:
                kind = ops[op_name]
                if kind in ('cbs', 'c3', 'sppf', 'concat', 'upsample', 'detect'):
                    continue  # Skip compound markers
                cur = _emit_single_op(c, ops, attrs, op_name, cur, inp, 'm')
            c.append('    layer[%d] = cur;' % i)
        
        save_comment = '  /* SAVED */' if i in saves else ''
        c.append('    /* layer[%d] done */%s' % (i, save_comment))
        c.append('')
    
    # Return last meaningful layer
    last_layer = max(routes.keys()) if routes else n_layers - 1
    c.append('    return layer[%d];' % last_layer)
    c.append('}')
    c.append('')
    
    # load_weights
    c.append('int load_weights(struct model * m, const char * path) {')
    c.append('    FILE * f = fopen(path, "rb");')
    c.append('    if (!f) { fprintf(stderr, "Cannot open %s\\n", path); return -1; }')
    c.append('    int n_tensors; fread(&n_tensors, 4, 1, f);')
    c.append('    printf("Loading %d tensors from %s\\n", n_tensors, path);')
    c.append('    for (int i = 0; i < n_tensors; i++) {')
    c.append('        int name_len; fread(&name_len, 4, 1, f);')
    c.append('        char name[256]; fread(name, 1, name_len, f);')
    c.append('        int ndims; fread(&ndims, 4, 1, f);')
    c.append('        int shape[4] = {1,1,1,1};')
    c.append('        for (int d = 0; d < ndims; d++) fread(&shape[d], 4, 1, f);')
    c.append('        int n_elems = 1;')
    c.append('        for (int d = 0; d < ndims; d++) n_elems *= shape[d];')
    c.append('        int n_bytes = n_elems * sizeof(float);')
    c.append('        struct ggml_tensor * target = NULL;')
    
    for name, kind in ops.items():
        if kind == 'conv2d':
            c.append('        if (strcmp(name, "%s_weight") == 0) target = m->%s_weight;' % (name, name))
        elif kind == 'batchnorm':
            for suffix in ['weight', 'bias', 'running_mean', 'running_var']:
                field = suffix.replace('running_', '')
                c.append('        if (strcmp(name, "%s_%s") == 0) target = m->%s_%s;' % (name, suffix, name, field))
    
    c.append('        if (target) { fread(target->data, 1, n_bytes, f); }')
    c.append('        else { fseek(f, n_bytes, SEEK_CUR); }')
    c.append('    }')
    c.append('    fclose(f);')
    c.append('    return 0;')
    c.append('}')
    c.append('')
    
    # main
    c.append('int main(int argc, char ** argv) {')
    c.append('    struct model m;')
    c.append('    model_init(&m);')
    c.append('    load_weights(&m, "yolov5n_weights.bin");')
    c.append('    struct ggml_init_params p = { .mem_size = 512*1024*1024, .mem_buffer = NULL, .no_alloc = false };')
    c.append('    struct ggml_context * ctx = ggml_init(p);')
    c.append('    struct ggml_tensor * input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 640, 640, 3, 1);')
    c.append('    ggml_set_name(input, "input");')
    c.append('    struct ggml_tensor * output = model_forward(&m, ctx, input);')
    c.append('    printf("Model built successfully\\n");')
    c.append('    ggml_free(ctx);')
    c.append('    ggml_free(m.ctx_w);')
    c.append('    return 0;')
    c.append('}')
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(c))
    print("Generated %s (%d lines)" % (output_file, len(c)))


if __name__ == '__main__':
    pl_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/auto_graph.pl'
    c_file = sys.argv[2] if len(sys.argv) > 2 else '/tmp/yolo_ggml_v2.c'
    ops, outputs, inputs, attrs, routes, saves, detect_inputs = parse_prolog(pl_file)
    emit_c(ops, outputs, inputs, attrs, routes, saves, detect_inputs, c_file)
