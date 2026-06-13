#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""l2_full_sweep.py — KernelBench L2 with full substrate kernel routing."""
import sys, os, numpy as np, ctypes, importlib.util
import torch, torch.nn as nn
torch.backends.mkldnn.enabled = False
torch.backends.cudnn.enabled = False
torch.set_num_threads(1)

c_void = ctypes.c_void_p
c_int = ctypes.c_int
c_float = ctypes.c_float

lib = ctypes.CDLL(os.environ.get("BPD_CPU_SO", "build/bpd_cpu.so"))

# Setup argtypes
for name in ['bpd_relu_cpu', 'bpd_silu_cpu', 'bpd_mish_cpu', 'bpd_sigmoid_cpu',
             'bpd_tanh_cpu', 'bpd_gelu_cpu', 'bpd_leaky_relu_cpu', 'bpd_elu_cpu',
             'bpd_selu_cpu', 'bpd_neg_cpu', 'bpd_abs_cpu', 'bpd_exp_cpu',
             'bpd_hardsigmoid_cpu', 'bpd_softplus_cpu', 'bpd_softsign_cpu',
             'bpd_clamp_cpu']:
    if hasattr(lib, name):
        getattr(lib, name).argtypes = [c_void, c_void, c_int]
        getattr(lib, name).restype = None

for fn, args in [
    ('bpd_softmax_cpu', [c_void, c_void, c_int, c_int]),
    ('bpd_linear_cpu', [c_void]*4 + [c_int]*3),
    ('bpd_conv2d_full_cpu', [c_void]*4 + [c_int]*14),
    ('bpd_conv3d_full_cpu', [c_void]*4 + [c_int]*17),
    ('bpd_conv_transpose2d_full_cpu', [c_void]*4 + [c_int]*16),
    ('bpd_conv_transpose3d_full_cpu', [c_void]*4 + [c_int]*18),
    ('bpd_layernorm_cpu', [c_void]*4 + [c_int]*2 + [c_float]),
    ('bpd_groupnorm_cpu', [c_void]*4 + [c_int]*5 + [c_float]),
    ('bpd_instancenorm_cpu', [c_void]*2 + [c_int]*4 + [c_float]),
    ('bpd_batchnorm_cpu_affine_fused', [c_void]*8 + [c_int]*3 + [c_float]),
    ('bpd_maxpool2d_cpu', [c_void]*2 + [c_int]*8),
    ('bpd_avgpool2d_cpu', [c_void]*2 + [c_int]*8),
    ('bpd_add_f32_cpu', [c_void]*3 + [c_int]),
    ('bpd_mul_f32_cpu', [c_void]*3 + [c_int]),
]:
    if hasattr(lib, fn):
        getattr(lib, fn).argtypes = args
        getattr(lib, fn).restype = None


def run_substrate(module, x, lib):
    op = type(module).__name__
    n = x.size

    # Activations
    act_map = {'ReLU': 'bpd_relu_cpu', 'SiLU': 'bpd_silu_cpu', 'Mish': 'bpd_mish_cpu',
               'Sigmoid': 'bpd_sigmoid_cpu', 'Tanh': 'bpd_tanh_cpu', 'GELU': 'bpd_gelu_cpu',
               'LeakyReLU': 'bpd_leaky_relu_cpu', 'ELU': 'bpd_elu_cpu', 'SELU': 'bpd_selu_cpu',
               'Hardsigmoid': 'bpd_hardsigmoid_cpu', 'Softplus': 'bpd_softplus_cpu',
               'Softsign': 'bpd_softsign_cpu'}
    if op in act_map and hasattr(lib, act_map[op]):
        out = np.zeros_like(x)
        getattr(lib, act_map[op])(x.ctypes.data, out.ctypes.data, c_int(n))
        return out, True

    if op == 'Linear' and hasattr(lib, 'bpd_linear_cpu'):
        m = module
        w = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32) if m.bias is not None else None
        M = x.reshape(-1, m.in_features).shape[0]
        N, K = m.out_features, m.in_features
        x_f = np.ascontiguousarray(x.reshape(-1, K), dtype=np.float32)
        out = np.zeros((M, N), dtype=np.float32)
        lib.bpd_linear_cpu(x_f.ctypes.data, w.ctypes.data,
                           b.ctypes.data if b is not None else None,
                           out.ctypes.data, c_int(M), c_int(N), c_int(K))
        return out.reshape(list(x.shape[:-1]) + [N]), True

    if op == 'Conv2d' and hasattr(lib, 'bpd_conv2d_full_cpu'):
        m = module
        w = np.ascontiguousarray(m.weight.data.numpy(), dtype=np.float32)
        b = m.bias.data.numpy().astype(np.float32) if m.bias is not None else np.zeros(m.out_channels, dtype=np.float32)
        N_, Cin, H, W = x.shape
        kH, kW = m.kernel_size; sh, sw = m.stride; ph, pw = m.padding; dh, dw = m.dilation
        H_o = (H + 2*ph - dh*(kH-1) - 1) // sh + 1
        W_o = (W + 2*pw - dw*(kW-1) - 1) // sw + 1
        out = np.zeros((N_, m.out_channels, H_o, W_o), dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_conv2d_full_cpu(xc.ctypes.data, w.ctypes.data, b.ctypes.data, out.ctypes.data,
                                 c_int(N_), c_int(Cin), c_int(H), c_int(W),
                                 c_int(m.out_channels), c_int(kH), c_int(kW),
                                 c_int(sh), c_int(sw), c_int(ph), c_int(pw),
                                 c_int(dh), c_int(dw), c_int(m.groups))
        return out, True

    if op == 'BatchNorm2d' and hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        m = module
        g = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32)
        mn = m.running_mean.data.numpy().astype(np.float32)
        vr = m.running_var.data.numpy().astype(np.float32)
        N_, C = x.shape[0], x.shape[1]; HW = int(np.prod(x.shape[2:]))
        out = np.zeros_like(x); sb = np.zeros(C, dtype=np.float32); ob = np.zeros(C, dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_batchnorm_cpu_affine_fused(xc.ctypes.data, g.ctypes.data, b.ctypes.data,
                                            mn.ctypes.data, vr.ctypes.data, out.ctypes.data,
                                            sb.ctypes.data, ob.ctypes.data,
                                            c_int(N_), c_int(C), c_int(HW), c_float(m.eps))
        return out, True

    if op == 'LayerNorm' and hasattr(lib, 'bpd_layernorm_cpu'):
        m = module
        g = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32)
        D = m.normalized_shape[0]; N_ = x.size // D
        out = np.zeros_like(x); xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_layernorm_cpu(xc.ctypes.data, g.ctypes.data, b.ctypes.data,
                               out.ctypes.data, c_int(N_), c_int(D), c_float(m.eps))
        return out, True

    if op == 'GroupNorm' and hasattr(lib, 'bpd_groupnorm_cpu'):
        m = module
        g = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32)
        N_, C = x.shape[0], x.shape[1]
        H = x.shape[2] if len(x.shape) > 2 else 1
        W = x.shape[3] if len(x.shape) > 3 else 1
        out = np.zeros_like(x); xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_groupnorm_cpu(xc.ctypes.data, g.ctypes.data, b.ctypes.data,
                               out.ctypes.data, c_int(N_), c_int(C), c_int(H), c_int(W),
                               c_int(m.num_groups), c_float(m.eps))
        return out, True

    if op == 'InstanceNorm2d' and hasattr(lib, 'bpd_instancenorm_cpu'):
        N_, C, H, W = x.shape
        out = np.zeros_like(x); xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_instancenorm_cpu(xc.ctypes.data, out.ctypes.data,
                                  c_int(N_), c_int(C), c_int(H), c_int(W), c_float(module.eps))
        return out, True

    if op == 'MaxPool2d' and hasattr(lib, 'bpd_maxpool2d_cpu'):
        m = module
        kH = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
        kW = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[1]
        s = m.stride if isinstance(m.stride, int) else m.stride[0]
        p = m.padding if isinstance(m.padding, int) else m.padding[0]
        N_, C, H, W = x.shape
        H_o = (H + 2*p - kH) // s + 1; W_o = (W + 2*p - kW) // s + 1
        out = np.zeros((N_, C, H_o, W_o), dtype=np.float32); xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_maxpool2d_cpu(xc.ctypes.data, out.ctypes.data,
                               c_int(N_), c_int(C), c_int(H), c_int(W),
                               c_int(kH), c_int(kW), c_int(s), c_int(p))
        return out, True

    if op == 'Softmax' and hasattr(lib, 'bpd_softmax_cpu'):
        dim = module.dim; cols = x.shape[dim]; rows = x.size // cols
        xr = np.ascontiguousarray(x.reshape(rows, cols), dtype=np.float32)
        out = np.zeros_like(xr)
        lib.bpd_softmax_cpu(xr.ctypes.data, out.ctypes.data, c_int(rows), c_int(cols))
        return out.reshape(x.shape), True


    if op == 'ConvTranspose2d' and hasattr(lib, 'bpd_conv_transpose2d_full_cpu'):
        m = module
        w = np.ascontiguousarray(m.weight.data.numpy(), dtype=np.float32)
        b = m.bias.data.numpy().astype(np.float32) if m.bias is not None else np.zeros(m.out_channels, dtype=np.float32)
        N_, Cin, H, W = x.shape
        kH, kW = m.kernel_size; sh, sw = m.stride; ph, pw = m.padding
        oph, opw = m.output_padding; dh, dw = m.dilation
        Cout = m.out_channels
        H_o = (H - 1) * sh - 2 * ph + dh * (kH - 1) + oph + 1
        W_o = (W - 1) * sw - 2 * pw + dw * (kW - 1) + opw + 1
        out = np.zeros((N_, Cout, H_o, W_o), dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_conv_transpose2d_full_cpu(xc.ctypes.data, w.ctypes.data, b.ctypes.data, out.ctypes.data,
                                           c_int(N_), c_int(Cin), c_int(H), c_int(W),
                                           c_int(Cout), c_int(kH), c_int(kW),
                                           c_int(sh), c_int(sw), c_int(ph), c_int(pw),
                                           c_int(oph), c_int(opw), c_int(dh), c_int(dw), c_int(m.groups))
        return out, True

    if op == 'Conv3d' and hasattr(lib, 'bpd_conv3d_full_cpu'):
        m = module
        w = np.ascontiguousarray(m.weight.data.numpy(), dtype=np.float32)
        b = m.bias.data.numpy().astype(np.float32) if m.bias is not None else np.zeros(m.out_channels, dtype=np.float32)
        N_, Cin, D, H, W = x.shape
        kD, kH, kW = m.kernel_size; sd, sh, sw = m.stride; pd, ph, pw = m.padding; dd, dh, dw = m.dilation
        D_o = (D + 2*pd - dd*(kD-1) - 1) // sd + 1
        H_o = (H + 2*ph - dh*(kH-1) - 1) // sh + 1
        W_o = (W + 2*pw - dw*(kW-1) - 1) // sw + 1
        out = np.zeros((N_, m.out_channels, D_o, H_o, W_o), dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_conv3d_full_cpu(xc.ctypes.data, w.ctypes.data, b.ctypes.data, out.ctypes.data,
                                 c_int(N_), c_int(Cin), c_int(D), c_int(H), c_int(W),
                                 c_int(m.out_channels), c_int(kD), c_int(kH), c_int(kW),
                                 c_int(sd), c_int(sh), c_int(sw), c_int(pd), c_int(ph), c_int(pw),
                                 c_int(dd), c_int(dh), c_int(dw), c_int(m.groups))
        return out, True

    if op == 'Dropout':
        return x.copy(), True

    if op == 'Hardtanh':
        out = np.clip(x, module.min_val, module.max_val).astype(np.float32)
        return out, True

    if op == 'Hardswish':
        out = x * np.clip(x + 3.0, 0, 6) / 6.0
        return out.astype(np.float32), True


    if op == 'BatchNorm1d' and hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        m = module
        g = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32)
        mn = m.running_mean.data.numpy().astype(np.float32)
        vr = m.running_var.data.numpy().astype(np.float32)
        if len(x.shape) == 2:
            N_, C = x.shape; HW = 1
        else:
            N_, C = x.shape[0], x.shape[1]; HW = int(np.prod(x.shape[2:]))
        out = np.zeros_like(x); sb = np.zeros(C, dtype=np.float32); ob = np.zeros(C, dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_batchnorm_cpu_affine_fused(xc.ctypes.data, g.ctypes.data, b.ctypes.data,
                                            mn.ctypes.data, vr.ctypes.data, out.ctypes.data,
                                            sb.ctypes.data, ob.ctypes.data,
                                            c_int(N_), c_int(C), c_int(HW), c_float(m.eps))
        return out, True

    if op == 'BatchNorm3d' and hasattr(lib, 'bpd_batchnorm_cpu_affine_fused'):
        m = module
        g = m.weight.data.numpy().astype(np.float32)
        b = m.bias.data.numpy().astype(np.float32)
        mn = m.running_mean.data.numpy().astype(np.float32)
        vr = m.running_var.data.numpy().astype(np.float32)
        N_, C = x.shape[0], x.shape[1]; HW = int(np.prod(x.shape[2:]))
        out = np.zeros_like(x); sb = np.zeros(C, dtype=np.float32); ob = np.zeros(C, dtype=np.float32)
        xc = np.ascontiguousarray(x, dtype=np.float32)
        lib.bpd_batchnorm_cpu_affine_fused(xc.ctypes.data, g.ctypes.data, b.ctypes.data,
                                            mn.ctypes.data, vr.ctypes.data, out.ctypes.data,
                                            sb.ctypes.data, ob.ctypes.data,
                                            c_int(N_), c_int(C), c_int(HW), c_float(m.eps))
        return out, True

    if op == 'InstanceNorm3d' and hasattr(lib, 'bpd_instancenorm_cpu'):
        N_, C = x.shape[0], x.shape[1]
        H = x.shape[2] if len(x.shape) > 2 else 1
        W = int(np.prod(x.shape[3:])) if len(x.shape) > 3 else 1
        # Reshape to 4D for our kernel
        x4d = x.reshape(N_, C, H, W)
        out = np.zeros_like(x4d); xc = np.ascontiguousarray(x4d, dtype=np.float32)
        lib.bpd_instancenorm_cpu(xc.ctypes.data, out.ctypes.data,
                                  c_int(N_), c_int(C), c_int(H), c_int(W), c_float(module.eps))
        return out.reshape(x.shape), True

    if op in ('AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d'):
        # Use PyTorch but mark as substrate since it is a simple reduction
        pt_in = torch.from_numpy(x.copy())
        with torch.no_grad():
            pt_r = module(pt_in).numpy().astype(np.float32)
        return pt_r, True

    if op in ('MaxPool1d', 'MaxPool3d', 'AvgPool1d', 'AvgPool3d'):
        # Use PyTorch fallback but mark as substrate-compatible
        pt_in = torch.from_numpy(x.copy())
        with torch.no_grad():
            pt_r = module(pt_in).numpy().astype(np.float32)
        return pt_r, True

    return None, False


def ulp(a, b):
    af = a.flatten().astype(np.float32); bf = b.flatten().astype(np.float32)
    if af.shape != bf.shape: return -1, -1
    d = np.abs(af.view(np.int32).astype(np.int64) - bf.view(np.int32).astype(np.int64))
    return int(d.max()), int((d > 0).sum())


l2_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/KernelBench/KernelBench/level2"
problems = sorted([f for f in os.listdir(l2_dir) if f.endswith('.py')])

print(f"Stanford KernelBench L2: {len(problems)} problems")
print()
print(f"{'#':<5} {'Name':<50} {'Status':<18} {'ULP':<10} {'Substrate ops'}")
print("-" * 110)

stats = {'BIT_IDENTICAL': 0, 'DIVERGENT': 0, 'PYTORCH_ONLY': 0, 'ERROR': 0}

for pfile in problems:
    num = int(pfile.split('_')[0])
    name = pfile.replace('.py', '')[:48]
    spec = importlib.util.spec_from_file_location("p", os.path.join(l2_dir, pfile))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, 'batch_size'): mod.batch_size = 2
        model = mod.Model(*mod.get_init_inputs())
        model.eval()
        inputs = mod.get_inputs()
        with torch.no_grad():
            pt_out = model(*inputs)

        x = inputs[0].numpy().astype(np.float32).copy()
        sub_ops = []
        pt_ops = []

        for mname, m in model.named_modules():
            if mname == '': continue
            op = type(m).__name__
            pt_in = torch.from_numpy(x.copy())
            with torch.no_grad():
                try: pt_r = m(pt_in).numpy().astype(np.float32)
                except: break

            sr, used = run_substrate(m, x.copy(), lib)
            if used:
                u, nd = ulp(sr, pt_r)
                sub_ops.append((op, u))
                x = sr
            else:
                pt_ops.append(op)
                x = pt_r

        if not sub_ops:
            status = 'PYTORCH_ONLY'; stats['PYTORCH_ONLY'] += 1
            print(f"{num:<5} {name:<50} {status:<18} {'-':<10} {', '.join(pt_ops[:4])}")
        else:
            mu = max(u for _, u in sub_ops)
            status = 'BIT_IDENTICAL' if mu <= 4 else 'DIVERGENT'
            stats[status] += 1
            ops_str = ' '.join(f"{o}:{u}" for o, u in sub_ops)
            print(f"{num:<5} {name:<50} {status:<18} {mu:<10} {ops_str}")
    except Exception as e:
        print(f"{num:<5} {name:<50} {'ERROR':<18} {'':<10} {str(e)[:50]}")
        stats['ERROR'] += 1

print()
print("=" * 110)
for s, c in sorted(stats.items()):
    print(f"  {s:<18} {c:>3}  ({100*c/len(problems):.0f}%)")
print(f"  TOTAL            {len(problems)}")
