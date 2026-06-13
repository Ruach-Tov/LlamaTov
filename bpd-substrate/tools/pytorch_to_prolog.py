#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""pytorch_to_prolog.py — Walk a PyTorch nn.Module and emit Prolog graph facts.

Usage:
    python pytorch_to_prolog.py model.pt output.pl

Emits:
    op_kind(node_name, bpd_op_kind).
    op_output(node_name, output_tensor_name).
    op_inputs(node_name, [input_tensor_names]).
    op_attr(node_name, attr_key, attr_value).

This replaces the hand-written yolo_graph.pl with auto-generated facts.
No cv2, PIL, or any non-torch dependency needed.
"""
import sys, torch, torch.nn as nn

# PyTorch type → BPD op_kind
TYPE_MAP = {
    'Conv2d':      'conv2d',
    'BatchNorm2d': 'batchnorm',
    'SiLU':        'silu',
    'ReLU':        'relu',
    'LeakyReLU':   'leaky_relu',
    'GELU':        'gelu',
    'Sigmoid':     'sigmoid',
    'Tanh':        'tanh',
    'MaxPool2d':   'maxpool',
    'AvgPool2d':   'avgpool',
    'Upsample':    'upsample',
    'Identity':    'identity',
    'Softmax':     'softmax',
    'LayerNorm':   'layernorm',
    'Linear':      'linear',
}

# Compound types with known forward() semantics
COMPOUND_MAP = {
    'Conv':       'cbs',        # Conv+BN+Act
    'Bottleneck': 'bottleneck',
    'C3':         'c3',
    'SPPF':       'sppf',
    'Concat':     'concat',
    'Detect':     'detect',
}


def sanitize(name):
    """Make a name Prolog-safe (lowercase atom)."""
    return name.replace('.', '_').replace('-', '_').lower()


def extract_attrs(mod, name):
    """Extract numeric attributes from a module."""
    attrs = []
    for key in ['in_channels', 'out_channels', 'kernel_size', 'stride',
                'padding', 'groups', 'eps', 'num_features', 'scale_factor']:
        if hasattr(mod, key):
            val = getattr(mod, key)
            if isinstance(val, tuple): val = val[0]
            attrs.append((sanitize(name), key, val))
    if hasattr(mod, 'add'):  # Bottleneck shortcut
        attrs.append((sanitize(name), 'shortcut', str(mod.add).lower()))
    return attrs


def walk_model(model, prefix='', parent_input=None):
    """Walk nn.Module tree, emit (facts, edges) for the compute graph."""
    facts = []
    attrs = []
    edges = []  # (from_output, to_input)
    
    layers = list(model.named_children())
    prev_output = parent_input  # data flows from parent input
    
    for i, (child_name, child_mod) in enumerate(layers):
        full_name = f"{prefix}_{child_name}" if prefix else child_name
        safe_name = sanitize(full_name)
        mod_type = type(child_mod).__name__
        
        # Determine BPD kind
        if mod_type in TYPE_MAP:
            bpd_kind = TYPE_MAP[mod_type]
        elif mod_type in COMPOUND_MAP:
            bpd_kind = COMPOUND_MAP[mod_type]
        else:
            bpd_kind = f"unknown_{mod_type.lower()}"
        
        output_name = f"{safe_name}_out"
        
        # Emit op_kind
        facts.append(f"op_kind({safe_name}, {bpd_kind})")
        facts.append(f"op_output({safe_name}, {output_name})")
        
        # Extract attributes
        attrs.extend(extract_attrs(child_mod, full_name))
        
        sub_children = list(child_mod.named_children())
        
        if not sub_children:
            # Leaf node: input from previous output
            if prev_output:
                facts.append(f"op_inputs({safe_name}, [{prev_output}])")
            prev_output = output_name
        else:
            # Compound node: recurse, then handle data flow
            if mod_type == 'Conv':
                # CBS: conv → bn → act, sequential chain
                sub_facts, sub_attrs, _ = walk_cbs(child_mod, full_name, prev_output)
                facts.extend(sub_facts)
                attrs.extend(sub_attrs)
                prev_output = output_name
                
            elif mod_type == 'Bottleneck':
                sub_facts, sub_attrs = walk_bottleneck(child_mod, full_name, prev_output)
                facts.extend(sub_facts)
                attrs.extend(sub_attrs)
                prev_output = output_name
                
            elif mod_type == 'C3':
                sub_facts, sub_attrs = walk_c3(child_mod, full_name, prev_output)
                facts.extend(sub_facts)
                attrs.extend(sub_attrs)
                prev_output = output_name
                
            elif mod_type == 'SPPF':
                sub_facts, sub_attrs = walk_sppf(child_mod, full_name, prev_output)
                facts.extend(sub_facts)
                attrs.extend(sub_attrs)
                prev_output = output_name
                
            elif mod_type == 'Concat':
                # Concat takes multiple inputs — handled at the model level
                facts.append(f"op_inputs({safe_name}, [multi])")
                prev_output = output_name
                
            elif mod_type == 'Detect':
                sub_facts, sub_attrs = walk_detect(child_mod, full_name, prev_output)
                facts.extend(sub_facts)
                attrs.extend(sub_attrs)
                prev_output = output_name
                
            else:
                # Generic: recurse
                sub_f, sub_a, _ = walk_model(child_mod, full_name, prev_output)
                facts.extend(sub_f)
                attrs.extend(sub_a)
                prev_output = output_name
    
    return facts, attrs, prev_output


def walk_cbs(mod, prefix, input_name):
    """Conv+BN+Act chain."""
    facts = []; attrs = []
    conv_name = sanitize(f"{prefix}_conv")
    bn_name = sanitize(f"{prefix}_bn")
    act_name = sanitize(f"{prefix}_act") if hasattr(mod, 'act') else None
    
    facts.append(f"op_kind({conv_name}, conv2d)")
    facts.append(f"op_output({conv_name}, {conv_name}_out)")
    facts.append(f"op_inputs({conv_name}, [{input_name}])")
    attrs.extend(extract_attrs(mod.conv, f"{prefix}_conv"))
    
    facts.append(f"op_kind({bn_name}, batchnorm)")
    facts.append(f"op_output({bn_name}, {bn_name}_out)")
    facts.append(f"op_inputs({bn_name}, [{conv_name}_out])")
    attrs.extend(extract_attrs(mod.bn, f"{prefix}_bn"))
    
    act_type = type(mod.act).__name__
    bpd_act = TYPE_MAP.get(act_type, act_type.lower())
    if act_name and bpd_act != 'identity':
        facts.append(f"op_kind({act_name}, {bpd_act})")
        facts.append(f"op_output({act_name}, {act_name}_out)")
        facts.append(f"op_inputs({act_name}, [{bn_name}_out])")
        last_out = f"{act_name}_out"
    else:
        last_out = f"{bn_name}_out"
    
    return facts, attrs, last_out


def walk_bottleneck(mod, prefix, input_name):
    """Bottleneck: cv1(CBS) → cv2(CBS) + optional residual add."""
    facts = []; attrs = []
    cv1_f, cv1_a, cv1_out = walk_cbs(mod.cv1, f"{prefix}_cv1", input_name)
    cv2_f, cv2_a, cv2_out = walk_cbs(mod.cv2, f"{prefix}_cv2", cv1_out)
    facts.extend(cv1_f + cv2_f)
    attrs.extend(cv1_a + cv2_a)
    
    if mod.add:
        add_name = sanitize(f"{prefix}_add")
        facts.append(f"op_kind({add_name}, add)")
        facts.append(f"op_output({add_name}, {add_name}_out)")
        facts.append(f"op_inputs({add_name}, [{cv2_out}, {input_name}])")
    
    return facts, attrs


def walk_c3(mod, prefix, input_name):
    """C3: cv1 → bottlenecks → concat(bn_out, cv2(input)) → cv3."""
    facts = []; attrs = []
    cv1_f, cv1_a, cv1_out = walk_cbs(mod.cv1, f"{prefix}_cv1", input_name)
    cv2_f, cv2_a, cv2_out = walk_cbs(mod.cv2, f"{prefix}_cv2", input_name)
    facts.extend(cv1_f + cv2_f)
    attrs.extend(cv1_a + cv2_a)
    
    # Walk bottlenecks
    bn_input = cv1_out
    for i, bn in enumerate(mod.m):
        bn_f, bn_a = walk_bottleneck(bn, f"{prefix}_bot{i}", bn_input)
        facts.extend(bn_f)
        attrs.extend(bn_a)
        if bn.add:
            bn_input = sanitize(f"{prefix}_bot{i}_add") + "_out"
        else:
            bn_input = sanitize(f"{prefix}_bot{i}_cv2_act") + "_out"
    
    # Concat
    concat_name = sanitize(f"{prefix}_concat")
    facts.append(f"op_kind({concat_name}, concat)")
    facts.append(f"op_output({concat_name}, {concat_name}_out)")
    facts.append(f"op_inputs({concat_name}, [{bn_input}, {cv2_out}])")
    
    # cv3
    cv3_f, cv3_a, cv3_out = walk_cbs(mod.cv3, f"{prefix}_cv3", f"{concat_name}_out")
    facts.extend(cv3_f)
    attrs.extend(cv3_a)
    
    return facts, attrs


def walk_sppf(mod, prefix, input_name):
    """SPPF: cv1 → pool → pool → pool → concat → cv2."""
    facts = []; attrs = []
    cv1_f, cv1_a, cv1_out = walk_cbs(mod.cv1, f"{prefix}_cv1", input_name)
    facts.extend(cv1_f); attrs.extend(cv1_a)
    
    pool_names = []
    prev = cv1_out
    for i in range(3):
        pn = sanitize(f"{prefix}_pool{i}")
        facts.append(f"op_kind({pn}, maxpool)")
        facts.append(f"op_output({pn}, {pn}_out)")
        facts.append(f"op_inputs({pn}, [{prev}])")
        attrs.append((pn, 'kernel_size', mod.m.kernel_size))
        pool_names.append(f"{pn}_out")
        prev = f"{pn}_out"
    
    concat_name = sanitize(f"{prefix}_concat")
    all_inputs = [cv1_out] + pool_names
    facts.append(f"op_kind({concat_name}, concat)")
    facts.append(f"op_output({concat_name}, {concat_name}_out)")
    facts.append(f"op_inputs({concat_name}, [{', '.join(all_inputs)}])")
    
    cv2_f, cv2_a, _ = walk_cbs(mod.cv2, f"{prefix}_cv2", f"{concat_name}_out")
    facts.extend(cv2_f); attrs.extend(cv2_a)
    
    return facts, attrs


def walk_detect(mod, prefix, input_name):
    """Detect: N conv heads."""
    facts = []; attrs = []
    for i, conv in enumerate(mod.m):
        cn = sanitize(f"{prefix}_head{i}")
        facts.append(f"op_kind({cn}, conv2d)")
        facts.append(f"op_output({cn}, {cn}_out)")
        attrs.extend(extract_attrs(conv, f"{prefix}_head{i}"))
    return facts, attrs


def emit_prolog(facts, attrs, filename):
    """Write Prolog file."""
    with open(filename, 'w') as f:
        f.write("%% Auto-generated by pytorch_to_prolog.py\n")
        f.write("%% Do not edit — regenerate from model checkpoint.\n")
        f.write(":- module(model_graph, [op_kind/2, op_output/2, op_inputs/2, op_attr/3]).\n\n")
        
        for fact in facts:
            f.write(f"{fact}.\n")
        
        f.write("\n%% Attributes\n")
        for name, key, val in attrs:
            if isinstance(val, float):
                f.write(f"op_attr({name}, {key}, {val}).\n")
            elif isinstance(val, bool):
                f.write(f"op_attr({name}, {key}, {str(val).lower()}).\n")
            else:
                f.write(f"op_attr({name}, {key}, {val}).\n")


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/yolo_canonical/yolov5n.pt'
    output_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/model_graph.pl'
    
    # Need yolov5 on path for class definitions during unpickle
    sys.path.insert(0, '/tmp/yolov5')
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().eval()
    
    facts, attrs, _ = walk_model(model)
    emit_prolog(facts, attrs, output_path)
    
    n_ops = sum(1 for f in facts if f.startswith('op_kind'))
    n_edges = sum(1 for f in facts if f.startswith('op_inputs'))
    print(f"Emitted {len(facts)} facts ({n_ops} ops, {n_edges} edges) + {len(attrs)} attrs")
    print(f"Written to {output_path}")
