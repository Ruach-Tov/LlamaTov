#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Verify Q4_K GPU dequant + measure tokens/sec vs CPU baseline.

Compares:
  1. GPU dequant vs CPU dequant (bit-identity)
  2. GPU qmatmul vs CPU qmatmul (bit-identity)
  3. Throughput: elements/sec for dequant, tokens/sec for matmul
"""
import ctypes, numpy as np, struct, sys, os, time

def build_libs():
    os.system("gcc -O2 -shared -fPIC -o /tmp/bpd_quant_cpu.so bench/bpd_quant.c -lm")
    ret = os.system("nvcc -O2 -shared -Xcompiler -fPIC -o /tmp/bpd_quant_gpu.so bench/bpd_quant_gpu.cu 2>/dev/null")
    return ret == 0

def ulp_max(a, b):
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    B = np.int64(0x80000000)
    ai = np.where(ai < 0, B - ai, ai)
    bi = np.where(bi < 0, B - bi, bi)
    return int(np.abs(ai - bi).max())

def main():
    has_gpu = build_libs()

    cpu = ctypes.CDLL("/tmp/bpd_quant_cpu.so")
    cpu.bpd_dequant_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cpu.bpd_qmatmul_q4k.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                      ctypes.c_int, ctypes.c_int]

    if has_gpu:
        gpu = ctypes.CDLL("/tmp/bpd_quant_gpu.so")
        gpu.bpd_dequant_q4k_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        gpu.bpd_qmatmul_q4k_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int, ctypes.c_int]
        gpu.qgpu_alloc.restype = ctypes.c_void_p
        gpu.qgpu_alloc.argtypes = [ctypes.c_int]
        gpu.qgpu_free.argtypes = [ctypes.c_void_p]
        gpu.qgpu_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        gpu.qgpu_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        gpu.qgpu_sync.argtypes = []

    # Load real Q4_K data from Mistral if available
    mistral_path = None
    for p in [
        "${OLLAMA_BLOBS:-~/.ollama/models/blobs}/sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f",
    ]:
        if os.path.exists(p):
            mistral_path = p
            break

    if mistral_path:
        print("Using real Mistral 7B Q4_K data")
        # Read a Q4_K tensor (blk.0.ffn_gate.weight)
        with open(mistral_path, "rb") as f:
            f.read(4)  # magic
            f.read(4)  # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n_kv):
                klen = struct.unpack("<Q", f.read(8))[0]; f.read(klen)
                vtype = struct.unpack("<I", f.read(4))[0]
                if vtype == 8:
                    slen = struct.unpack("<Q", f.read(8))[0]; f.read(slen)
                elif vtype in (4,5,6): f.read(4)
                elif vtype == 7: f.read(1)
                elif vtype == 10: f.read(8)
                elif vtype == 9:
                    etype = struct.unpack("<I", f.read(4))[0]
                    count = struct.unpack("<Q", f.read(8))[0]
                    if etype == 8:
                        for _ in range(count):
                            sl = struct.unpack("<Q", f.read(8))[0]; f.read(sl)
                    elif etype in (0,1,7): f.read(count)
                    elif etype in (4,5,6): f.read(count*4)
                    elif etype in (10,11): f.read(count*8)
                    elif etype in (2,3): f.read(count*2)
            tensors = []
            for _ in range(n_tensors):
                nlen = struct.unpack("<Q", f.read(8))[0]
                name = f.read(nlen).decode()
                ndims = struct.unpack("<I", f.read(4))[0]
                dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
                ttype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]
                tensors.append((name, dims, ttype, offset))
            meta_end = f.tell()
            data_start = meta_end + ((32 - meta_end % 32) % 32)

            # Find ffn_gate (Q4_K, 4096x14336)
            for name, dims, ttype, offset in tensors:
                if 'ffn_gate' in name and ttype == 12:
                    n_elements = dims[0] * dims[1]
                    n_blocks = n_elements // 256
                    # Use first 1024 blocks (262144 elements) for testing
                    n_test = min(1024, n_blocks)
                    f.seek(data_start + offset)
                    qdata = np.frombuffer(f.read(n_test * 144), dtype=np.uint8).copy()
                    print("  Tensor: %s  dims=%s" % (name, dims))
                    print("  Testing %d blocks (%d elements)" % (n_test, n_test * 256))
                    break
    else:
        print("No Mistral model found — using synthetic Q4_K data")
        n_test = 1024
        qdata = np.random.default_rng(42).integers(0, 256, n_test * 144, dtype=np.uint8)

    K = n_test * 256
    M = 16  # simulate 16 output rows for matmul

    # === CPU Dequant ===
    print("\n=== CPU Dequant ===")
    cpu_output = np.zeros(K, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        cpu.bpd_dequant_q4k(qdata.ctypes.data, cpu_output.ctypes.data, n_test)
    t1 = time.perf_counter()
    cpu_dequant_ms = (t1 - t0) / 10 * 1000
    print("  Time: %.2f ms (%d elements)" % (cpu_dequant_ms, K))
    print("  Throughput: %.1f M elements/sec" % (K / cpu_dequant_ms / 1000))

    if not has_gpu:
        print("\nNo GPU available — skipping GPU tests")
        return

    # === GPU Dequant ===
    print("\n=== GPU Dequant ===")
    d_qdata = gpu.qgpu_alloc(len(qdata))
    d_output = gpu.qgpu_alloc(K * 4)
    gpu.qgpu_h2d(d_qdata, qdata.ctypes.data, len(qdata))
    gpu.qgpu_sync()

    # Warmup
    gpu.bpd_dequant_q4k_gpu(d_qdata, d_output, n_test)
    gpu.qgpu_sync()

    t0 = time.perf_counter()
    for _ in range(100):
        gpu.bpd_dequant_q4k_gpu(d_qdata, d_output, n_test)
    gpu.qgpu_sync()
    t1 = time.perf_counter()
    gpu_dequant_ms = (t1 - t0) / 100 * 1000
    print("  Time: %.3f ms (%d elements)" % (gpu_dequant_ms, K))
    print("  Throughput: %.1f M elements/sec" % (K / gpu_dequant_ms / 1000))
    print("  Speedup vs CPU: %.1fx" % (cpu_dequant_ms / gpu_dequant_ms))

    # Verify GPU matches CPU
    gpu_output = np.zeros(K, dtype=np.float32)
    gpu.qgpu_d2h(gpu_output.ctypes.data, d_output, K * 4)
    ulp = ulp_max(cpu_output, gpu_output)
    print("  GPU vs CPU ULP: %d" % ulp)

    # === Quantized Matmul ===
    print("\n=== Quantized Matmul (M=%d, K=%d) ===" % (M, K))
    x = np.random.default_rng(42).standard_normal(K).astype(np.float32)
    
    # CPU matmul
    # Build M rows of quantized data (reuse same data M times for testing)
    qweight = np.tile(qdata, M)
    cpu_mv = np.zeros(M, dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(10):
        cpu.bpd_qmatmul_q4k(qweight.ctypes.data, x.ctypes.data,
                              cpu_mv.ctypes.data, M, K)
    t1 = time.perf_counter()
    cpu_mv_ms = (t1 - t0) / 10 * 1000
    print("  CPU: %.2f ms" % cpu_mv_ms)

    # GPU matmul
    d_qweight = gpu.qgpu_alloc(len(qweight))
    d_x = gpu.qgpu_alloc(K * 4)
    d_mv = gpu.qgpu_alloc(M * 4)
    gpu.qgpu_h2d(d_qweight, qweight.ctypes.data, len(qweight))
    gpu.qgpu_h2d(d_x, x.ctypes.data, K * 4)
    gpu.qgpu_sync()

    gpu.bpd_qmatmul_q4k_gpu(d_qweight, d_x, d_mv, M, K)
    gpu.qgpu_sync()

    t0 = time.perf_counter()
    for _ in range(100):
        gpu.bpd_qmatmul_q4k_gpu(d_qweight, d_x, d_mv, M, K)
    gpu.qgpu_sync()
    t1 = time.perf_counter()
    gpu_mv_ms = (t1 - t0) / 100 * 1000
    print("  GPU: %.3f ms" % gpu_mv_ms)
    print("  Speedup: %.1fx" % (cpu_mv_ms / gpu_mv_ms))

    # Verify
    gpu_mv = np.zeros(M, dtype=np.float32)
    gpu.qgpu_d2h(gpu_mv.ctypes.data, d_mv, M * 4)
    mv_ulp = ulp_max(cpu_mv, gpu_mv)
    print("  GPU vs CPU ULP: %d" % mv_ulp)

    # Cleanup
    gpu.qgpu_free(d_qdata)
    gpu.qgpu_free(d_output)
    gpu.qgpu_free(d_qweight)
    gpu.qgpu_free(d_x)
    gpu.qgpu_free(d_mv)

    print("\n=== Summary ===")
    print("  Dequant: CPU %.2f ms, GPU %.3f ms (%.1fx speedup)" % (cpu_dequant_ms, gpu_dequant_ms, cpu_dequant_ms/gpu_dequant_ms))
    print("  Matmul:  CPU %.2f ms, GPU %.3f ms (%.1fx speedup)" % (cpu_mv_ms, gpu_mv_ms, cpu_mv_ms/gpu_mv_ms))
    print("  GPU vs CPU: dequant %d ULP, matmul %d ULP" % (ulp, mv_ulp))

if __name__ == "__main__":
    main()
