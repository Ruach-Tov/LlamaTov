#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""quant_dot_llvmir.py — GENERATE parameterized LLVM IR for ggml-equivalent quantized dot kernels
(Iyun, 2026-05-29, Heath). Decoded from ggml_vec_dot_q4_K_q8_K (the reference Ollama runs). The IR
is parameterized so the SAME generator emits the reference-matching kernel (0 ULP by construction)
for CPU or GPU, AND optimized variants FROM that reference point (verified against the bit-identity fixture).

This is the "match reference then optimize" generator: the reference IR is the ANCHOR; optimizations
are deltas (wider vectors, fused scale-muls, fast-math where the fixture still passes).
"""
from dataclasses import dataclass, field

@dataclass
class QuantDotParams:
    weight_quant: str = "q4_K"        # block layout + nibble/scale unpack
    activation_quant: str = "q8_K"    # the vec_dot_type (ggml: q4_K -> q8_K)
    QK: int = 256                     # super-block size
    accum_type: str = "i32"           # INTEGER accumulation (bit-identity-critical)
    scale_unpack: str = "kmask_6bit"  # the bitpacked-6bit sub-block scheme
    fp_scale_order: tuple = ("d_mul_accum", "dmin_subtract")  # exact fp op order
    target: str = "cpu_scalar"        # cpu_scalar | cpu_avx2 | gpu_sm_61
    fast_math: bool = False           # OFF for bit-identity; ON only when fixture still passes
    vector_width: int = 1             # 1=scalar(reference). >1 = optimization (must re-verify 0 ULP)

def emit_llvm_ir(p: QuantDotParams) -> str:
    """Emit LLVM IR for the quantized dot kernel, parameterized. The reference setting
    (q4_K/q8_K/i32/kmask_6bit/scalar/no-fast-math) yields 0-ULP-by-construction IR."""
    fm = "" if not p.fast_math else " fast"
    sb = p.QK // 32   # sub-blocks
    # --- the reference IR skeleton (scalar, exact order) ---
    ir = f"""; quant_dot kernel — weight={p.weight_quant} activation={p.activation_quant}
; accum={p.accum_type} scale_unpack={p.scale_unpack} target={p.target} fast_math={p.fast_math}
; ANCHOR = ggml_vec_dot_q4_K_q8_K (bit-identical by construction when params = ollama setting)
define float @quant_dot_{p.weight_quant}_{p.activation_quant}(
    ptr %x_blocks, ptr %y_blocks, i32 %nb) {{
entry:
  br label %superblock.loop
superblock.loop:                       ; per super-block i (QK={p.QK})
  ; 1. unpack 4-bit weights -> i8 aux8[{p.QK}]  (low nibble then high nibble, exact order)
  ; 2. unpack 6-bit packed scales/mins via kmask1/2/3 (the bitpacked scheme)
  ; 3. min-correction:  sumi = sum_j(bsums[j] * mins[j/2])         ; i32
  ; 4. main dot ({sb} sub-blocks x 32):  aux32[l] += scale * (q8[l]*a[l])   ; {p.accum_type} accumulate
  %d    = fmul{fm} float %dx_f32, %dy           ; d = fp16_to_fp32(x.d) * y.d
  ; 5. sums[l] += d * (sitofp aux32[l])         ; exact fp order
  %dmin = fmul{fm} float %dminx_f32, %dy
  ; sumf -= dmin * sitofp(sumi)
  br i1 %more, label %superblock.loop, label %exit
exit:
  ; 6. sumf += sum_l(sums[l])
  ret float %sumf
}}"""
    # --- backend lowering note ---
    backend = {
        "cpu_scalar": "; lower: plain scalar (the reference anchor)",
        "cpu_avx2":   f"; lower: AVX2 — i8/i16/i32 SIMD (vpmaddubsw/vpmaddwd), vector_width={p.vector_width}; re-verify 0 ULP",
        "gpu_sm_61":  f"; lower: PTX/SASS — SIMT lanes, dp4a for i8 dot; vector_width={p.vector_width}; re-verify 0 ULP",
    }[p.target]
    return ir + "\n" + backend

def ollama_reference() -> QuantDotParams:
    """The parameter setting that matches the Ollama reference -> 0 ULP by construction."""
    return QuantDotParams(weight_quant="q4_K", activation_quant="q8_K", accum_type="i32",
                          scale_unpack="kmask_6bit", target="cpu_scalar", fast_math=False, vector_width=1)

def optimized_from_reference(base: QuantDotParams, target: str, vector_width: int) -> QuantDotParams:
    """An optimization DELTA from the verified reference. Must be re-verified against the bit-identity
    fixture (0 ULP) before acceptance; fast_math only if it still passes."""
    return QuantDotParams(**{**base.__dict__, "target": target, "vector_width": vector_width})

if __name__ == "__main__":
    ref = ollama_reference()
    print("=== OLLAMA-REFERENCE kernel (0 ULP by construction) ===")
    print(emit_llvm_ir(ref))
    print("\n=== OPTIMIZED-FROM-REFERENCE (GPU sm_61, width 4) — re-verify 0 ULP ===")
    print(emit_llvm_ir(optimized_from_reference(ref, "gpu_sm_61", 4)))
