# ggml_vec_dot_q4_K_q8_K — DECODED reference algorithm → parameterized LLVM IR

Decoded from the lifted llama.cpp scalar reference (ggml-cpu-quants.c:7640-7694) + cross-checked
against the x86 disassembly (ggml-cpu-quants.c.o <ggml_vec_dot_q4_K_q8_K>). This is THE exact
computation Ollama runs for a Q4_K-weight × fp32-activation matmul. Bit-identity = replicate THIS.
(Iyun, 2026-05-29, Heath.)

## BLOCK STRUCTURE (the operands)
- block_q4_K (weight, 144 bytes, QK_K=256 values): { half d; half dmin; uint8 scales[12]; uint8 qs[128] }
    - d, dmin: fp16 super-block scale + min
    - scales[12]: 8 sub-block 6-bit scales + 8 sub-block 6-bit mins, BITPACKED (the kmask unpacking)
    - qs[128]: 256 4-bit weight quants (2 per byte)
- block_q8_K (activation, quantized from fp32, QK_K=256): { float d; int8 qs[256]; int16 bsums[16] }
    - d: fp32 activation block scale
    - qs[256]: int8 activation quants
    - bsums[16]: per-16 sums of qs (for the min correction)

## THE ALGORITHM (per super-block i, exact order = the bit-identity contract)
1. UNPACK 4-bit weights to int8 aux8[256]: for each of 4 groups of 64:
     a[0..31]  = q4[l] & 0xF      (low nibbles)
     a[32..63] = q4[l] >> 4       (high nibbles)
2. UNPACK the bitpacked 6-bit scales/mins (kmask1=0x3f3f3f3f, kmask2=0x0f0f0f0f, kmask3=0x03030303):
     utmp[3] = ((utmp[2]>>4)&kmask2) | (((utmp[1]>>6)&kmask3)<<4)
     uaux    = utmp[1] & kmask1
     utmp[1] = (utmp[2]&kmask2) | (((utmp[0]>>6)&kmask3)<<4)
     utmp[2] = uaux ; utmp[0] &= kmask1
     scales = (uint8*)&utmp[0]   (8 sub-block scales)
     mins   = (uint8*)&utmp[2]   (8 sub-block mins)
3. MIN CORRECTION (the dmin term): sumi = Σ_{j=0..15} y.bsums[j] * mins[j/2]
4. MAIN DOT (8 sub-blocks of 32, int16 mul → int32 accumulate, scaled per sub-block):
     for j in 0..7:  scale = scales[j]
       for 4 chunks of 8:  aux16[l] = q8[l]*a[l] ; aux32[l] += scale * aux16[l]   (int32 accum)
5. SCALE TO FLOAT (the exact fp order):
     d    = fp16_to_fp32(x.d) * y.d        ; sums[l] += d * aux32[l]   (l=0..7)
     dmin = fp16_to_fp32(x.dmin) * y.d     ; sumf -= dmin * sumi
6. FINAL: sumf += Σ_{l=0..7} sums[l]

## THE PARAMETERS (what makes this a GENERATOR, not one kernel)
quant_dot_kernel(
    weight_quant   = q4_K,        % {q4_0,q4_1,q4_K,q5_K,q6_K,q8_0,...} -> block layout + unpack
    activation_quant = q8_K,      % the vec_dot_type (ggml: q4_K -> q8_K)
    QK             = 256,         % super-block size
    sub_blocks     = 8,          % QK/32
    accum_type     = int32,       % integer accumulation (NOT fp32 — this is what my naive test missed)
    scale_unpack   = kmask_6bit,  % the bitpacked-6bit scheme (vs simple per-block)
    fp_scale_order = [d_mul_then_accum, dmin_subtract],  % EXACT fp op order (bit-identity-critical)
    target         = {cpu_avx2, cpu_scalar, gpu_sm_61}   % backend
)
Each parameter selects a piece of the generated IR. weight_quant=q4_K + activation_quant=q8_K +
accum_type=int32 + scale_unpack=kmask_6bit is the OLLAMA setting -> 0 ULP by construction.

## WHY MY EARLIER NAIVE TEST FAILED (the lesson)
I did round-to-int8 + dequantize-back + fp32-matmul. WRONG on 3 counts the decode reveals:
1. accumulation is INTEGER (int32 aux32), not fp32 — different rounding
2. scales are 6-bit bitpacked sub-block (8 per super-block), not 1 scale per 256
3. the fp scale-and-min-correction has a SPECIFIC op order (d*aux32 sum, then -dmin*sumi)
Reading the reference removes the guessing: replicate THESE and it is 0 ULP.

## LLVM IR GENERATION (CPU or GPU, match-then-optimize)
The decoded algorithm maps to LLVM IR we GENERATE, parameterized by the above:
- integer loop nest (steps 1-4) -> i8/i16/i32 vector ops; on CPU -> AVX2 intrinsics, on GPU -> SIMT lanes
- the fp scale (step 5) -> fmul/fadd in the EXACT order (fast-math OFF to preserve bit-identity)
- match: emit IR equivalent to ggml's -> 0 ULP + tick-comparable (same op count/order)
- optimize FROM there: once 0-ULP-verified, relax (e.g. fuse the two scale muls, vectorize wider,
  fast-math where the fixture still passes) -> faster, with the fixture proving we stayed correct.
The reference IR is the ANCHOR; optimizations are deltas verified against the bit-identity fixture.
