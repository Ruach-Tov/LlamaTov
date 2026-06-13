#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
llamatov_gpu.py — GPU inference via our own CUDA kernels on P4 (sm_61).

Loads GGUF weights, transfers to GPU via ctypes, runs forward pass
through our BPD-generated kernel library. PyTorch not needed for GPU ops.
"""
import ctypes, numpy as np, struct, time, sys, os

# ═══════════════════════════════════════════════════════════════
# CUDA KERNEL LIBRARY (compiled from llamatov_kernels.cu)
# ═══════════════════════════════════════════════════════════════

NVIDIA_LIB = '/nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib'
CUDA_LIB = '/nix/store/560i0agldlr2h4h3bx6mq2lifw6w1iaa-cuda-native-redist-12.8/lib'
os.environ['LD_LIBRARY_PATH'] = f'{NVIDIA_LIB}:{CUDA_LIB}:' + os.environ.get('LD_LIBRARY_PATH', '')

lib = ctypes.CDLL("/tmp/llamatov_kernels.so")

# Function signatures
lib.gpu_alloc.restype = ctypes.c_void_p
lib.gpu_alloc.argtypes = [ctypes.c_int]
lib.gpu_free.argtypes = [ctypes.c_void_p]
lib.gpu_copy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gpu_copy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.gpu_matmul.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
lib.gpu_add.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
lib.gpu_mul.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]
lib.gpu_silu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
lib.gpu_gelu.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]
lib.gpu_rms_norm.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int, ctypes.c_int, ctypes.c_float]
lib.gpu_rms_norm.restype = None
lib.gpu_scale.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_float, ctypes.c_int]
lib.gpu_sync.argtypes = []

class GPUTensor:
    """Wrapper around a GPU memory pointer with shape info."""
    def __init__(self, shape, data=None):
        self.shape = tuple(shape)
        self.size = 1
        for d in shape: self.size *= d
        self.bytes = self.size * 4
        self.ptr = lib.gpu_alloc(self.bytes)
        if data is not None:
            assert data.size == self.size, f"Size mismatch: {data.size} vs {self.size}"
            flat = data.astype(np.float32).flatten()
            lib.gpu_copy_h2d(self.ptr, flat.ctypes.data, self.bytes)
    
    def to_numpy(self):
        out = np.zeros(self.size, dtype=np.float32)
        lib.gpu_copy_d2h(out.ctypes.data, self.ptr, self.bytes)
        return out.reshape(self.shape)
    
    def free(self):
        if self.ptr: lib.gpu_free(self.ptr); self.ptr = None
    
    @property
    def rows(self): return self.shape[0] if len(self.shape) >= 2 else 1
    
    @property
    def cols(self): return self.shape[-1]

def gpu_zeros(shape):
    t = GPUTensor(shape)
    # Zero-initialize by copying zeros
    z = np.zeros(t.size, dtype=np.float32)
    lib.gpu_copy_h2d(t.ptr, z.ctypes.data, t.bytes)
    return t

# ═══════════════════════════════════════════════════════════════
# GPU OPS
# ═══════════════════════════════════════════════════════════════

def matmul(a, b, out):
    """out[M,N] = a[M,K] @ b[K,N]"""
    M = a.rows; K = a.cols; N = b.cols
    lib.gpu_matmul(a.ptr, b.ptr, out.ptr, M, N, K)

def add_inplace(a, b):
    """a += b (element-wise)"""
    lib.gpu_add(a.ptr, b.ptr, a.ptr, a.size)

def add_to(a, b, out):
    """out = a + b"""
    lib.gpu_add(a.ptr, b.ptr, out.ptr, a.size)

def mul_inplace(a, b):
    """a *= b (element-wise)"""
    lib.gpu_mul(a.ptr, b.ptr, a.ptr, a.size)

def silu_inplace(a):
    lib.gpu_silu(a.ptr, a.ptr, a.size)

def gelu_inplace(a):
    lib.gpu_gelu(a.ptr, a.ptr, a.size)

def rms_norm(inp, weight, out, eps=1e-5):
    rows = inp.rows; cols = inp.cols
    lib.gpu_rms_norm(inp.ptr, weight.ptr, out.ptr, rows, cols, ctypes.c_float(eps))

def scale_inplace(a, s):
    lib.gpu_scale(a.ptr, a.ptr, ctypes.c_float(s), a.size)

def copy_gpu(src, dst):
    """GPU-to-GPU copy."""
    assert src.bytes == dst.bytes
    # Use H2D with src pointer (it's already device memory — this is a hack)
    # Better: add gpu_copy_d2d to the kernel library
    # For now, round-trip through CPU
    tmp = src.to_numpy()
    lib.gpu_copy_h2d(dst.ptr, tmp.ctypes.data, dst.bytes)

# ═══════════════════════════════════════════════════════════════
# GGUF LOADING (reuse from llamatov_run.py)
# ═══════════════════════════════════════════════════════════════

def parse_gguf(path):
    # Import the parser from our existing runner
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llamatov_run import parse_gguf as _parse, lt as _lt
    return _parse(path), _lt

def load_weights_to_gpu(path):
    """Parse GGUF, dequantize all tensors, load to GPU."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llamatov_run import parse_gguf, lt
    
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    cfg = {
        'arch': arch,
        'n_layers': md.get(f'{arch}.block_count', 12),
        'n_head': md.get(f'{arch}.attention.head_count', 12),
        'n_head_kv': md.get(f'{arch}.attention.head_count_kv',
                     md.get(f'{arch}.attention.head_count', 12)),
        'n_embd': md.get(f'{arch}.embedding_length', 768),
        'norm_eps': md.get(f'{arch}.attention.layer_norm_rms_epsilon',
                    md.get(f'{arch}.attention.layer_norm_epsilon', 1e-5)),
    }
    
    print(f"Arch: {arch}, layers: {cfg['n_layers']}, heads: {cfg['n_head']}/{cfg['n_head_kv']}, embd: {cfg['n_embd']}")
    
    # Load and dequantize on CPU, then transfer to GPU
    w = {}
    for name, info in ts.items():
        cpu_tensor = lt(path, do, info).numpy()
        w[name] = GPUTensor(cpu_tensor.shape, cpu_tensor)
    
    return w, cfg

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/llamatov-data/model-zoo/gpt2-124m-Q2_K.gguf'
    
    print(f"=== LlamaTov GPU Inference ===")
    print(f"Model: {path}")
    
    t0 = time.time()
    w, cfg = load_weights_to_gpu(path)
    t1 = time.time()
    print(f"Loaded {len(w)} tensors to GPU in {t1-t0:.1f}s")
    
    print(f"\nWeights on GPU. Ready for inference.")
    print(f"Architecture: {cfg['arch']}")
    print(f"Next: implement forward pass through GPU ops")
