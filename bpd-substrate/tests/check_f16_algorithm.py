# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""check_f16_algorithm.py — compare our f16_to_f32 vs ggml_compute_fp16_to_fp32.

ggml uses the XNNPACK magic-number trick:
  two_w = w + w  (w = h << 16)
  normalized_value = fp32_from_bits((two_w >> 4) + 0xE0000000) * 0x1.0p-112f
  denormalized_value = fp32_from_bits((two_w >> 17) | (126 << 23)) - 0.5f
  result = sign | (two_w < (1 << 27) ? denormalized_value : normalized_value)

Our f16_to_f32 uses the standard sign/exponent/mantissa decomposition.
Both should be IEEE 754 correct, but we need to verify they produce identical
F32 bits for ALL 65536 F16 values.
"""
import struct
import numpy as np

def ggml_fp16_to_fp32(h):
    """Mirror of ggml_compute_fp16_to_fp32 from ggml-impl.h."""
    h = int(h) & 0xFFFF
    w = (h << 16) & 0xFFFFFFFF
    sign = w & 0x80000000
    two_w = (w + w) & 0xFFFFFFFF
    exp_offset = (0xE0 << 23) & 0xFFFFFFFF

    # exp_scale = 0x1.0p-112f = 2^-112
    exp_scale_bits = 0x7800000  # (127 - 112) << 23 = 15 << 23
    exp_scale = struct.unpack('f', struct.pack('I', exp_scale_bits))[0]

    norm_bits = ((two_w >> 4) + exp_offset) & 0xFFFFFFFF
    normalized_value = struct.unpack('f', struct.pack('I', norm_bits))[0] * exp_scale

    magic_mask = (126 << 23) & 0xFFFFFFFF
    magic_bias = 0.5
    denorm_bits = ((two_w >> 17) | magic_mask) & 0xFFFFFFFF
    denormalized_value = struct.unpack('f', struct.pack('I', denorm_bits))[0] - magic_bias

    denormalized_cutoff = 1 << 27
    if two_w < denormalized_cutoff:
        result_bits = sign | struct.unpack('I', struct.pack('f', denormalized_value))[0]
    else:
        result_bits = sign | struct.unpack('I', struct.pack('f', normalized_value))[0]

    return struct.unpack('f', struct.pack('I', result_bits & 0xFFFFFFFF))[0]

def our_f16_to_f32(h):
    """Mirror of our bpd_cpu.c f16_to_f32 — standard decomposition."""
    h = int(h) & 0xFFFF
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0x1F:
        # Inf or NaN
        f32_bits = (sign << 31) | 0x7F800000 | (mant << 13)
        return struct.unpack('f', struct.pack('I', f32_bits))[0]
    elif exp == 0:
        if mant == 0:
            # Zero
            return -0.0 if sign else 0.0
        else:
            # Subnormal F16 -> normal F32
            # Find leading bit
            e = -1
            m = mant
            while m:
                e += 1
                m >>= 1
            # e is the position of the leading bit (0-indexed from LSB)
            # The value is: (-1)^sign * 2^(-14) * (mant / 1024)
            # = (-1)^sign * 2^(-14) * mant * 2^(-10)
            # = (-1)^sign * mant * 2^(-24)
            # Normalize: mant * 2^(-24) = (mant << (10-e)) * 2^(-24-10+e) / 1024
            # F32 exponent: -14 - (10 - e) + 127 = 103 + e  (biased)
            f32_exp = 103 + e
            # Mantissa: shift mant to have leading 1 at bit 10, then shift to 23 bits
            f32_mant = (mant << (13 - e)) & 0x7FFFFF
            f32_bits = (sign << 31) | (f32_exp << 23) | f32_mant
            return struct.unpack('f', struct.pack('I', f32_bits))[0]
    else:
        # Normal F16
        f32_exp = exp - 15 + 127
        f32_mant = mant << 13
        f32_bits = (sign << 31) | (f32_exp << 23) | f32_mant
        return struct.unpack('f', struct.pack('I', f32_bits))[0]

# Compare all 65536 F16 values
n_diff = 0
diffs = []
for h in range(65536):
    ggml_val = ggml_fp16_to_fp32(h)
    our_val = our_f16_to_f32(h)
    if struct.pack('f', ggml_val) != struct.pack('f', our_val):
        n_diff += 1
        if len(diffs) < 10:
            diffs.append((h, ggml_val, our_val))

print(f"Total F16 values: 65536")
print(f"Differences: {n_diff}")
if diffs:
    print("First 10 differences:")
    for h, ggml_val, our_val in diffs:
        exp = (h >> 10) & 0x1F
        mant = h & 0x3FF
        print(f"  h=0x{h:04x} (exp={exp}, mant={mant}): ggml={ggml_val:.8e}  ours={our_val:.8e}")
else:
    print("PERFECT MATCH: both algorithms produce identical F32 bits for all 65536 F16 values")

# Also check numpy's f16->f32 conversion
print()
print("Checking numpy f16->f32 vs ggml:")
n_diff_np = 0
for h in range(65536):
    u16 = np.uint16(h)
    np_val = float(u16.view(np.float16))
    ggml_val = ggml_fp16_to_fp32(h)
    if struct.pack('f', np_val) != struct.pack('f', ggml_val):
        n_diff_np += 1
print(f"numpy vs ggml differences: {n_diff_np}")
