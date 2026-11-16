#!/usr/bin/env python3
"""
p4_run_harness.py — THE RUN: load the pre-built verified cubins on the P4,
launch each, verify GPU-side bit-exactness at the kink, and measure speedup
vs the Level-0 memory floor.

Author: Iyun (2026-11-16). Pounce-prep — the one-command benchmark for when the
P4 frees. All 4 cubins are PRE-BUILT offline (oneway/strict/leaky_backward +
relu_forward, sm_61, bit-exact-verified vs torch on CPU 0-ULP). This harness is
the RUN step: cuModuleLoad + launch + GPU-side verify + timing.

CONTRACT (what this measures, honestly):
  1. GPU-side correctness: launch each cubin, compare vs torch-GPU at the kink.
     - BACKWARD: expect 0-ULP vs torch-GPU (Medayek measured: backward uniform
       across ±0 on BOTH devices — so GPU should match CPU-verified kernels).
     - FORWARD: expect the DOCUMENTED divergence — our kernel preserves -0
       (ge-passthrough), torch-GPU CLEARS -0 (sign bit 0). This is the §6
       forward-only divergence, NAMED not a bug. Report it as EXPECTED.
  2. Speedup vs Level-0 floor: time the fused kernel, compare against the
     noop_copy memory-floor baseline (probe_results.lgt). The fused kernel's
     time MINUS the Level-0 floor = the actual compute cost. Report the measured
     speedup vs eager (separate-kernel) torch, Amdahl-framed:
       - this net: ~5% (conv GEMMs dominate) — report AS measured, 5% is 5%.
       - EXTRAPOLATED to wider nets (activation-heavy): marked EXTRAPOLATED.

USAGE (on the enclave, system python3 with cuda=True, when the P4 is free):
  the system Python with CUDA-torch (python3 on most systems; /run/current-system/sw/bin/python3 on NixOS) p4_run_harness.py [--n 1000000] [--iters 100]

NOTE: requires CUDA-torch (enclave system python3) + the pre-built cubins in cwd.
This is a SKELETON pending the THE-CARD-IS-YOURS ping — the cubins are ready;
this wires the load+launch+verify+time. Review by Bocher before the live run.
"""
import sys, os, struct, time, argparse, ctypes

def bits(x):
    import numpy as np
    return struct.pack("<f", float(x)).hex()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="element count")
    ap.add_argument("--iters", type=int, default=100, help="timing iterations")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch not importable. Run on enclave system python3 (cuda=True)."); sys.exit(1)
    if not torch.cuda.is_available():
        print("FATAL: cuda unavailable. This must run on the P4 (enclave system python3)."); sys.exit(1)
    print(f"  device: {torch.cuda.get_device_name(0)} | arch_list: {torch.cuda.get_arch_list()}")

    import numpy as np
    # pycuda or torch's cuModuleLoad path — use torch.cuda + a cubin loader.
    # Skeleton: torch doesn't directly cuModuleLoad; use cuda-python (cuda.cuda) if present,
    # else fall back to comparing an EQUIVALENT torch impl (correctness) + timing torch eager.
    try:
        from cuda import cuda, cudart  # cuda-python
        have_cuda_py = True
    except ImportError:
        have_cuda_py = False
        print("  NOTE: cuda-python not present — cubin cuModuleLoad needs it. "
              "Correctness verify falls back to torch-equivalent; timing uses torch eager as baseline.")

    cubins = ["relu_forward.cubin","strict_backward.cubin","leaky_backward.cubin","oneway_backward.cubin"]
    for cb in cubins:
        if not os.path.exists(cb):
            print(f"  MISSING cubin: {cb} (pre-build it offline first)"); 

    # --- Kink verification vectors (Medayek's edge-seeded set) ---
    xs = torch.tensor([0.0,-0.0,1e-30,-1e-30,2.5,-2.5], dtype=torch.float32, device="cuda")
    gs = torch.tensor([1.0,-1.0,0.5,-0.5,2.0,-3.0], dtype=torch.float32, device="cuda")

    print("\n=== GPU-SIDE CORRECTNESS (kink verify vs torch-GPU) ===")
    # FORWARD: expect documented -0 divergence (our kernel preserves, torch-GPU clears)
    fwd_torch = torch.relu(xs)
    print(f"  forward torch-GPU relu(-0.0) = {bits(fwd_torch[1].cpu())} "
          f"({'(+0, GPU CLEARS -0 — the DOCUMENTED divergence; our kernel PRESERVES -0)' if bits(fwd_torch[1].cpu())=='00000000' else '(-0 preserved)'})")
    print("    => §6 convention: our kernel preserves -0 (CPU-oracle-aligned); torch-GPU clears. "
          "Forward-only, backward-unaffected. EXPECTED, not a bug.")
    # BACKWARD: expect 0-ULP vs torch-GPU (uniform across ±0 on both devices)
    # torch strict backward = where(x>0, g, 0):
    bwd_strict = torch.where(xs>0, gs, torch.zeros_like(gs))
    print(f"  backward strict torch-GPU at dead side: {[bits(v.cpu()) for v in bwd_strict]} "
          "(expect our where-style cubin matches 0-ULP)")

    print("\n=== SPEEDUP vs LEVEL-0 FLOOR (Amdahl-framed) ===")
    print("  Level-0 noop_copy floor: see probe_results.lgt (memory floor, P4 ~192GB/s).")
    print("  TODO (live): time fused cubin vs eager torch; subtract Level-0 floor = compute cost.")
    print("  HONEST FRAMING: ~5% of THIS net (conv GEMMs dominate) — report 5% AS 5%;")
    print("  big-perf EXTRAPOLATED to activation-heavy nets (marked EXTRAPOLATED).")

    print("\n  [SKELETON — pending THE-CARD-IS-YOURS + Bocher review of the cuModuleLoad path]")

if __name__ == "__main__":
    main()
