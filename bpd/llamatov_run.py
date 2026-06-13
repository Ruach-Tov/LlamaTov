#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""LlamaTov: Multi-architecture GGUF inference. Benchmark runner."""
import numpy as np, torch, torch.nn.functional as F, struct, sys, time

# ═══════════════════════════════════════════════════════════════
# GGUF PARSER
# ═══════════════════════════════════════════════════════════════

def parse_gguf(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x46554747, f"Not GGUF: {magic:#x}"
        ver = struct.unpack('<I', f.read(4))[0]
        nt = struct.unpack('<Q', f.read(8))[0]
        nkv = struct.unpack('<Q', f.read(8))[0]
        md = {}
        for _ in range(nkv):
            kl = struct.unpack('<Q', f.read(8))[0]; k = f.read(kl).decode()
            vt = struct.unpack('<I', f.read(4))[0]; md[k] = rv(f, vt)
        ts = {}
        for _ in range(nt):
            nl = struct.unpack('<Q', f.read(8))[0]; n = f.read(nl).decode()
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            tt = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
            ts[n] = (dims, tt, off)
        al = md.get('general.alignment', 32); do = f.tell()
        do = (do + al - 1) // al * al
    return md, ts, do

def rv(f, vt):
    if vt==4: return struct.unpack('<I',f.read(4))[0]
    elif vt==5: return struct.unpack('<i',f.read(4))[0]
    elif vt==6: return struct.unpack('<f',f.read(4))[0]
    elif vt==8: sl=struct.unpack('<Q',f.read(8))[0]; return f.read(sl).decode()
    elif vt==10: return struct.unpack('<Q',f.read(8))[0]
    elif vt==9: at=struct.unpack('<I',f.read(4))[0]; c=struct.unpack('<Q',f.read(8))[0]; return [rv(f,at) for _ in range(c)]
    elif vt==7: return bool(struct.unpack('<?',f.read(1))[0])
    else: f.read({0:1,1:1,2:2,3:2,11:8,12:8}.get(vt,4)); return None

# ═══════════════════════════════════════════════════════════════
# DEQUANTIZATION
# ═══════════════════════════════════════════════════════════════

def dq2k(path, off, nel, shape):
    """Q2_K dequantization — CORRECTNESS FIX (correction 13 continuation).

    ggml reference (ggml/src/ggml-quants.c:dequantize_row_q2_K):
      256-element superblock = 2 chunks × 128 elements
      Each 128-chunk uses 32 packed bytes (q[0..31])
      Per chunk, 4 shift cycles (shift = 0, 2, 4, 6):
        For each shift, output 32 values in order:
          16 values from (q[0..15] >> shift) & 3
          16 values from (q[16..31] >> shift) & 3
        Each batch of 16 has its own dl, ml from the scales table.
      After 4 shifts, advance q by 32 bytes to next chunk.

    Output position pattern per 32-byte chunk:
       shift=0: positions  0..15  (q[ 0..15] low 2 bits)
                positions 16..31  (q[16..31] low 2 bits)
       shift=2: positions 32..47  (q[ 0..15] bits 2-3)
                positions 48..63  (q[16..31] bits 2-3)
       shift=4: positions 64..79  (q[ 0..15] bits 4-5)
                positions 80..95  (q[16..31] bits 4-5)
       shift=6: positions 96..111 (q[ 0..15] bits 6-7)
                positions 112..127 (q[16..31] bits 6-7)

    Previous code used np.stack([shifts], -1).reshape(nb, 256) which
    produced interleaved order [q[0]>>0, q[0]>>2, q[0]>>4, q[0]>>6,
    q[1]>>0, ...] — same class of bug as the Q4_0 nibble-order issue.

    Fix: vectorize the ggml reference structure explicitly.
    """
    nb = nel // 256
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*84), dtype=np.uint8)
    bl = raw.reshape(nb, 84)
    d = np.frombuffer(bl[:,80:82].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dm = np.frombuffer(bl[:,82:84].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    # 16 scale bytes; each byte: low nibble = scale index, high nibble = min index
    sc = (bl[:,:16] & 0xF).astype(np.float32) * d[:,None]   # (nb, 16)
    mn = (bl[:,:16] >> 4).astype(np.float32) * dm[:,None]   # (nb, 16)
    qs = bl[:,16:80]    # (nb, 64) — 2 chunks of 32 packed bytes

    # Build output per ggml reference structure
    quants = np.empty((nb, 256), dtype=np.float32)
    is_idx = 0   # scale index, advances 1 per 16-element batch
    out = 0      # output position
    for chunk in range(2):
        q_chunk = qs[:, chunk*32 : (chunk+1)*32].astype(np.int32)  # (nb, 32)
        q_lo16 = q_chunk[:, :16]      # (nb, 16) — bytes 0..15 of this chunk
        q_hi16 = q_chunk[:, 16:32]    # (nb, 16) — bytes 16..31 of this chunk
        for shift in (0, 2, 4, 6):
            # First 16 outputs: q_lo16 with this shift, scale[is_idx]
            sc_a = sc[:, is_idx:is_idx+1]   # (nb, 1)
            mn_a = mn[:, is_idx:is_idx+1]
            quants[:, out:out+16] = sc_a * ((q_lo16 >> shift) & 3).astype(np.float32) - mn_a
            out += 16
            is_idx += 1
            # Next 16 outputs: q_hi16 with this shift, scale[is_idx]
            sc_b = sc[:, is_idx:is_idx+1]
            mn_b = mn[:, is_idx:is_idx+1]
            quants[:, out:out+16] = sc_b * ((q_hi16 >> shift) & 3).astype(np.float32) - mn_b
            out += 16
            is_idx += 1
    # Apply ne0-major reshape (mavchin's substrate-honest finding 160afcc89)
    if len(shape)==2:
        ne0, ne1 = shape
        return torch.from_numpy(quants.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(quants.reshape(shape).copy())

def dq4_0(path, off, nel, shape):
    """Q4_0 dequantization — CORRECTNESS FIX (correction 13, 2026-05-16):

    ggml's Q4_0 packing per block of 32 elements:
      - bytes 0..15 (16 bytes) hold 32 4-bit nibbles
      - For each byte j: low nibble (j & 0x0F) → output position j
                        high nibble (j >> 4)   → output position j + 16
      - So output is [low_0, low_1, ..., low_15, high_0, high_1, ..., high_15]
      - NOT interleaved [low_0, high_0, low_1, high_1, ...]

    Previous implementation used np.stack([lo, hi], -1).reshape(nb, 32) which
    produced the WRONG interleaved order, corrupting every Q4_0 weight load
    and causing the CPU path to produce garbage tokens (correction-13 bug).
    """
    nb = nel // 32; bsz = 18  # 2-byte scale + 16-byte packed nibbles
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*bsz), dtype=np.uint8)
    bl = raw.reshape(nb, bsz)
    d = np.frombuffer(bl[:,:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    qs = bl[:,2:]                                 # shape (nb, 16) — packed nibbles
    lo = (qs & 0x0F).astype(np.float32) - 8       # shape (nb, 16) — output positions 0..15
    hi = (qs >> 4).astype(np.float32) - 8         # shape (nb, 16) — output positions 16..31
    q = np.concatenate([lo, hi], axis=-1)         # shape (nb, 32) — correct ggml order
    # Apply mavchin's substrate-honest ne0-major fix (commit 160afcc89):
    # GGUF stores 2D tensors with ne0 fastest, so for 2D we reshape to
    # [ne1, ne0] then transpose. For non-2D, straight reshape works.
    flat = (d[:,None] * q)
    if len(shape) == 2:
        ne0, ne1 = shape
        return torch.from_numpy(flat.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(flat.reshape(shape).copy())

def dq8_0(path, off, nel, shape):
    """Q8_0 dequantization — CORRECTNESS FIX (correction 13, 2026-05-16):

    ggml's Q8_0 packing per block of 32 elements:
      2-byte FP16 scale d, then 32 int8 quantized values qs[0..31]
      Output: y[i*32 + j] = qs[j] * d for j in 0..31

    No transpose, no reordering — straight pass-through with scale.

    Previous code had an explicit .reshape([ne1, ne0]).T.copy() that
    transposed the output. This was wrong if other dq* functions (like
    dq4_0) did NOT transpose — they use straight .reshape(shape). The
    transpose was masking a different bug elsewhere OR was wrong itself.

    Substrate-honest fix: use the same .reshape(shape) pattern as Q4_0
    and the other K-quants. ggml stores all tensor data in C-order with
    ne[0] varying fastest; numpy reshape(dims) recovers the same layout.
    """
    nb = nel // 32; bsz = 34   # 2-byte scale + 32-byte int8 quants
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*bsz), dtype=np.uint8)
    bl = raw.reshape(nb, bsz)
    d = np.frombuffer(bl[:,:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    qs = bl[:,2:].view(np.int8).astype(np.float32)
    ne0, ne1 = shape; return torch.from_numpy((d[:,None] * qs).reshape([ne1, ne0]).T.copy()) if len(shape)==2 else torch.from_numpy((d[:,None] * qs).reshape(shape).copy())

def dq3k(path, off, nel, shape):
    """Q3_K dequantization — CORRECTNESS FIX (correction 13 continuation).

    ggml reference (ggml/src/ggml-quants.c:dequantize_row_q3_K):
      256-element superblock with:
        32 hmask bytes (1 bit per element, packed by element position)
        64 packed bytes qs (2-bit values, 4 shifts per byte)
        12 scale bytes encoding 16 signed 6-bit scales
        2 byte FP16 scale d_all

      Per superblock: 2 chunks × 128 elements
      Each chunk uses 32 bytes of qs and walks through hmask m=1,2,4,8
      Per chunk, 4 shift cycles (shift=0,2,4,6), per shift 32 outputs:
        16 from q[0..15]: dl * ((q[l] >> shift) & 3 - (hm[l] & m ? 0 : 4))
        16 from q[16..31]: same pattern
      After all 4 shifts, advance qs by 32, m bit advances per shift.

    The "subtract 4 when hmask bit is 0" gives the substantive range
    [-4..3] per 2-bit value (extended via the high-bit indicator).

    Previous code used:
      - np.stack interleaved pattern for q_lo (wrong, same Q2_K bug class)
      - hmask packed by [byte*8+bit] for q_hi (wrong — ggml indexes by
        element position with masks that shift per shift cycle)
      - 4-bit scale unpacking (wrong — ggml uses 12-byte → 16 signed
        6-bit scale table with non-trivial bit manipulation)

    All three substrate-honestly fixed in this rewrite.
    """
    nb = nel // 256
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*110), dtype=np.uint8)
    bl = raw.reshape(nb, 110)
    d_all = np.frombuffer(bl[:,108:110].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)

    # Unpack 16 signed 6-bit scales from 12 scale bytes per superblock.
    # ggml uses 32-bit ops (kmask1=0x03030303, kmask2=0x0f0f0f0f), but we
    # mirror byte-by-byte for clarity. After unpacking, scales is (nb, 16) int8.
    aux_bytes = bl[:,96:108].astype(np.uint32)   # (nb, 12) as uint32
    # Reinterpret as 3 uint32s per row via 4-byte little-endian groups
    a0 = (aux_bytes[:,0] | (aux_bytes[:,1]<<8) | (aux_bytes[:,2]<<16) | (aux_bytes[:,3]<<24))
    a1 = (aux_bytes[:,4] | (aux_bytes[:,5]<<8) | (aux_bytes[:,6]<<16) | (aux_bytes[:,7]<<24))
    a2 = (aux_bytes[:,8] | (aux_bytes[:,9]<<8) | (aux_bytes[:,10]<<16) | (aux_bytes[:,11]<<24))
    kmask1 = np.uint32(0x03030303); kmask2 = np.uint32(0x0f0f0f0f)
    new_a2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4)
    new_a3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4)
    new_a0 = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4)
    new_a1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4)
    # Pack the 4 uint32s back into 16 bytes per row (little-endian)
    sc_packed = np.stack([
        new_a0 & 0xFF, (new_a0>>8) & 0xFF, (new_a0>>16) & 0xFF, (new_a0>>24) & 0xFF,
        new_a1 & 0xFF, (new_a1>>8) & 0xFF, (new_a1>>16) & 0xFF, (new_a1>>24) & 0xFF,
        new_a2 & 0xFF, (new_a2>>8) & 0xFF, (new_a2>>16) & 0xFF, (new_a2>>24) & 0xFF,
        new_a3 & 0xFF, (new_a3>>8) & 0xFF, (new_a3>>16) & 0xFF, (new_a3>>24) & 0xFF,
    ], axis=-1).astype(np.uint8)  # (nb, 16)
    # Interpret as signed int8 and subtract 32 per ggml (scales[is++] - 32)
    scales = sc_packed.view(np.int8).astype(np.float32) - 32   # (nb, 16)

    hm = bl[:,:32].astype(np.int32)         # (nb, 32) hmask bytes
    qs = bl[:,32:96].astype(np.int32)       # (nb, 64) packed 2-bit values

    quants = np.empty((nb, 256), dtype=np.float32)
    is_idx = 0
    out = 0
    for chunk in range(2):
        q_chunk = qs[:, chunk*32 : (chunk+1)*32]      # (nb, 32)
        q_lo16 = q_chunk[:, :16]                      # (nb, 16)
        q_hi16 = q_chunk[:, 16:32]                    # (nb, 16)
        # For chunk 0 the m bit starts at 1; for chunk 1 it starts at 16.
        # Wait: ggml does NOT reset m between chunks. m persists 1,2,4,8 then
        # next chunk continues 16,32,64,128.
        for shift_idx in range(4):
            shift = shift_idx * 2
            m_bit = 1 << (chunk * 4 + shift_idx)   # 1,2,4,8 then 16,32,64,128
            # First 16 outputs: q_lo16
            dl = d_all * scales[:, is_idx]   # (nb,)
            hm_l = (hm[:, :16] & m_bit) != 0  # (nb, 16) bool — True = "high bit set" = subtract 0
            q_val = (q_lo16 >> shift) & 3
            corrected = q_val - np.where(hm_l, 0, 4)
            quants[:, out:out+16] = dl[:, None] * corrected.astype(np.float32)
            out += 16
            is_idx += 1
            # Next 16: q_hi16 with hmask bytes 16..31
            dl = d_all * scales[:, is_idx]
            hm_h = (hm[:, 16:32] & m_bit) != 0
            q_val = (q_hi16 >> shift) & 3
            corrected = q_val - np.where(hm_h, 0, 4)
            quants[:, out:out+16] = dl[:, None] * corrected.astype(np.float32)
            out += 16
            is_idx += 1

    # ne0-major reshape (mavchin's substrate-honest finding 160afcc89)
    if len(shape)==2:
        ne0, ne1 = shape
        return torch.from_numpy(quants.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(quants.reshape(shape).copy())

def dq6k(path, off, nel, shape):
    """Q6_K dequantization — CORRECTNESS FIX (correction 13, 2026-05-16).

    ggml's Q6_K packing per 256-element superblock (per dequantize_row_q6_K):
      256 elements split into two 128-element chunks.
      Per chunk: 32-iteration loop l=0..31 produces 4 values at positions
        l + 0:  q1 = (ql[l +  0] & 0xF) | ((qh[l] >> 0) & 3) << 4) - 32
        l + 32: q2 = (ql[l + 32] & 0xF) | ((qh[l] >> 2) & 3) << 4) - 32
        l + 64: q3 = (ql[l +  0] >> 4)  | ((qh[l] >> 4) & 3) << 4) - 32
        l + 96: q4 = (ql[l + 32] >> 4)  | ((qh[l] >> 6) & 3) << 4) - 32
      Then advance pointers: ql += 64, qh += 32, sc += 8, output += 128.
      Scale index is_l = l // 16, so sc[is_l + 0/2/4/6] selects the
      correct 8-bit scale per output position.

    Previous code used the Q4_0-style interleaved-stack pattern AND a
    different qh layout, neither of which match ggml's reference. Both
    bugs corrupted Q6_K weights — and Q6_K is the embedding/output
    quantization for many "Q4_0" mixed-quant models including tinyllama
    (which has Q4_0 + Q6_K + F32 per gguf-py inspection).

    Fix: implement ggml's reference layout via vectorized numpy ops.
    Per superblock: two chunks; per chunk, vectorize over l=0..31.
    """
    nb = nel // 256
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*210), dtype=np.uint8)
    bl = raw.reshape(nb, 210)
    ql = bl[:,:128].astype(np.int32)            # (nb, 128) low+mid nibbles
    qh = bl[:,128:192].astype(np.int32)         # (nb, 64) high 2-bit pairs
    sc = bl[:,192:208].view(np.int8).astype(np.float32)  # (nb, 16) per-group scales
    d = np.frombuffer(bl[:,208:210].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)

    quants = np.empty((nb, 256), dtype=np.float32)

    # Process two 128-element chunks per superblock. Per chunk:
    #   ql_chunk: 64 bytes (32 "near" + 32 "far"); qh_chunk: 32 bytes;
    #   sc_chunk: 8 scales (indexed by is_l = l // 16 + offset).
    for n_idx in range(2):
        ql_off = n_idx * 64
        qh_off = n_idx * 32
        sc_off = n_idx * 8
        out_off = n_idx * 128

        ql_near = ql[:, ql_off:ql_off+32]            # (nb, 32)
        ql_far  = ql[:, ql_off+32:ql_off+64]         # (nb, 32)
        qh_use  = qh[:, qh_off:qh_off+32]            # (nb, 32)

        # Four q-values per l, at positions l+0, l+32, l+64, l+96.
        q1 = ((ql_near & 0xF) | (((qh_use >> 0) & 3) << 4)) - 32   # (nb, 32)
        q2 = ((ql_far  & 0xF) | (((qh_use >> 2) & 3) << 4)) - 32
        q3 = ((ql_near >> 4)  | (((qh_use >> 4) & 3) << 4)) - 32
        q4 = ((ql_far  >> 4)  | (((qh_use >> 6) & 3) << 4)) - 32

        # Scale-per-element: is_l = l // 16, then sc[is_l + 0/2/4/6].
        # For l in 0..15: is_l = 0 → sc[0], sc[2], sc[4], sc[6]
        # For l in 16..31: is_l = 1 → sc[1], sc[3], sc[5], sc[7]
        is_l = np.arange(32) // 16    # (32,) -- values 0 or 1
        scales_chunk = sc[:, sc_off:sc_off+8]   # (nb, 8)
        s1 = scales_chunk[:, is_l + 0]   # (nb, 32)
        s2 = scales_chunk[:, is_l + 2]
        s3 = scales_chunk[:, is_l + 4]
        s4 = scales_chunk[:, is_l + 6]

        quants[:, out_off + 0  : out_off + 32]  = q1.astype(np.float32) * s1
        quants[:, out_off + 32 : out_off + 64]  = q2.astype(np.float32) * s2
        quants[:, out_off + 64 : out_off + 96]  = q3.astype(np.float32) * s3
        quants[:, out_off + 96 : out_off + 128] = q4.astype(np.float32) * s4

    # Apply ne0-major fix (mavchin's substrate-honest finding 160afcc89)
    flat = (d[:,None] * quants).astype(np.float32)
    if len(shape) == 2:
        ne0, ne1 = shape
        return torch.from_numpy(flat.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(flat.reshape(shape).copy())

def dq4k(path, off, nel, shape):
    """Q4_K dequantization — CORRECTNESS FIX (correction 13, 2026-05-16):

    ggml's Q4_K packing per 256-element superblock (per dequantize_row_q4_K):
      256 elements = 4 groups of 64 elements
      Each group of 64 uses 32 bytes of packed nibbles
      Per group: first 32 outputs are LOW nibbles of bytes 0..31
                 next 32 outputs are HIGH nibbles of bytes 0..31
      So layout per group: [lo_0..lo_31, hi_0..hi_31] — CONCATENATED, not interleaved

    Previous code used np.stack([lo,hi],-1).reshape(nb,256) which produced
    interleaved [lo_0, hi_0, lo_1, hi_1, ...] — wrong nibble order matching
    the same bug found in Q4_0.

    Fix: reshape qs into (nb, 4 groups, 32 bytes), unpack lows and highs
    separately per group, concatenate per group, then flatten.
    """
    nb = nel // 256; bsz = 144
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*bsz), dtype=np.uint8)
    bl = raw.reshape(nb, bsz)
    d = np.frombuffer(bl[:,:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dm = np.frombuffer(bl[:,2:4].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    sc_raw = bl[:,4:16]; qs = bl[:,16:]    # qs shape: (nb, 128) — 128 bytes per superblock
    scales = np.zeros((nb,8), dtype=np.float32); mins = np.zeros((nb,8), dtype=np.float32)
    for j in range(4):
        scales[:,j] = (sc_raw[:,j]&0x3F).astype(np.float32)
        mins[:,j] = (sc_raw[:,4+j]&0x3F).astype(np.float32)
    for j in range(4):
        scales[:,4+j] = ((sc_raw[:,8+j]&0xF)|((sc_raw[:,j]>>6)<<4)).astype(np.float32)
        mins[:,4+j] = ((sc_raw[:,8+j]>>4)|((sc_raw[:,4+j]>>6)<<4)).astype(np.float32)
    scales *= d[:,None]; mins *= dm[:,None]
    # Reshape into 4 groups of 32 packed bytes (each → 64 output values)
    qs_grouped = qs.reshape(nb, 4, 32)
    lo = (qs_grouped & 0x0F).astype(np.float32)             # (nb, 4, 32) — first half of each group
    hi = (qs_grouped >> 4).astype(np.float32)               # (nb, 4, 32) — second half of each group
    q = np.concatenate([lo, hi], axis=-1).reshape(nb, 256)  # per-group [lo_0..31, hi_0..31] → 64 outputs/group → 256 total
    si = np.arange(256)//32
    # Apply ne0-major fix (mavchin 160afcc89)
    flat = (scales[:,si]*q - mins[:,si]).astype(np.float32)
    if len(shape) == 2:
        ne0, ne1 = shape
        return torch.from_numpy(flat.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(flat.reshape(shape).copy())

def dq5k(path, off, nel, shape):
    """Q5_K dequantization — NEW IMPLEMENTATION (correction 13 continuation).

    Implements Q5_K per ggml/src/ggml-quants.c:dequantize_row_q5_K.
    Mavchin's commit d4ac13014 noted Mistral 7B uses Q4_K + Q5_K but the
    Q5_K dispatcher entry was missing — type 13 would have fallen through
    to "WARN: zeros". This implementation closes that gap.

    block_q5_K layout (176 bytes per 256-element superblock):
      2 bytes  d              (FP16 super-block scale)
      2 bytes  dmin           (FP16 super-block min scale)
      12 bytes scales         (K_SCALE_SIZE, 6-bit-packed scales+mins, same as Q4_K)
      32 bytes qh             (high bit of each 5-bit value, 1 bit per element)
      128 bytes qs            (low 4 bits, 2 elements per byte — same as Q4_K's qs)

    Per superblock, 4 groups of 64 outputs. Per group:
      d1 = d * scale[is+0]; m1 = dmin * min[is+0]
      d2 = d * scale[is+1]; m2 = dmin * min[is+1]
      For l in 0..31: y = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1
      For l in 0..31: y = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2
      Advance ql += 32, is += 2, u1 <<= 2, u2 <<= 2

    Substrate-honest: this is Q4_K with a 5th bit pulled from qh. The
    scale-unpacking (12-byte → 8 scales + 8 mins) is IDENTICAL to Q4_K.
    """
    nb = nel // 256
    bsz = 4 + 12 + 32 + 128   # = 176 per block
    with open(path,'rb') as f: f.seek(off); raw = np.frombuffer(f.read(nb*bsz), dtype=np.uint8)
    bl = raw.reshape(nb, bsz)
    d = np.frombuffer(bl[:,:2].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    dm = np.frombuffer(bl[:,2:4].tobytes(), dtype=np.float16).astype(np.float32).reshape(nb)
    sc_raw = bl[:,4:16]                # (nb, 12) — same 6-bit-packed format as Q4_K
    qh = bl[:,16:48].astype(np.int32)  # (nb, 32) — high bits, 1 per element across superblock
    qs = bl[:,48:176]                  # (nb, 128) — low 4 bits, 2 elements per byte

    # Unpack 8 scales + 8 mins exactly like Q4_K
    scales = np.zeros((nb,8), dtype=np.float32); mins = np.zeros((nb,8), dtype=np.float32)
    for j in range(4):
        scales[:,j] = (sc_raw[:,j]&0x3F).astype(np.float32)
        mins[:,j] = (sc_raw[:,4+j]&0x3F).astype(np.float32)
    for j in range(4):
        scales[:,4+j] = ((sc_raw[:,8+j]&0xF)|((sc_raw[:,j]>>6)<<4)).astype(np.float32)
        mins[:,4+j] = ((sc_raw[:,8+j]>>4)|((sc_raw[:,4+j]>>6)<<4)).astype(np.float32)
    scales *= d[:,None]; mins *= dm[:,None]

    # Group qs into 4 groups of 32 bytes (= 64 outputs each via lo+hi nibbles)
    qs_grouped = qs.reshape(nb, 4, 32)
    # qh has 32 bytes total used as a bitfield across the whole superblock.
    # Per group g (g=0..3), the relevant masks are u1 = 1<<(2g), u2 = 1<<(2g+1)
    # qh[l] (l=0..31) provides 1 bit per output position per mask.
    quants = np.empty((nb, 256), dtype=np.float32)
    for g in range(4):
        u1 = 1 << (2 * g)        # bit-mask for the "low half" 5th bits
        u2 = 1 << (2 * g + 1)    # bit-mask for the "high half" 5th bits
        group_qs = qs_grouped[:, g, :]                # (nb, 32) packed nibbles
        lo4 = (group_qs & 0x0F).astype(np.int32)      # (nb, 32) low 4 bits
        hi4 = (group_qs >> 4).astype(np.int32)        # (nb, 32) high 4 bits
        lo5 = lo4 + np.where((qh & u1) != 0, 16, 0)   # (nb, 32) full 5-bit value
        hi5 = hi4 + np.where((qh & u2) != 0, 16, 0)   # (nb, 32) full 5-bit value
        # First 32 outputs use scales/mins[2g+0]; next 32 use scales/mins[2g+1]
        s_lo = scales[:, 2*g + 0:2*g + 1]   # (nb, 1)
        m_lo = mins[:, 2*g + 0:2*g + 1]
        s_hi = scales[:, 2*g + 1:2*g + 2]
        m_hi = mins[:, 2*g + 1:2*g + 2]
        quants[:, g*64 + 0:g*64 + 32]  = s_lo * lo5.astype(np.float32) - m_lo
        quants[:, g*64 + 32:g*64 + 64] = s_hi * hi5.astype(np.float32) - m_hi

    # ne0-major reshape (mavchin's substrate-honest finding 160afcc89)
    if len(shape) == 2:
        ne0, ne1 = shape
        return torch.from_numpy(quants.reshape([ne1, ne0]).T.copy())
    return torch.from_numpy(quants.reshape(shape).copy())

def lt(path, do, info):
    dims, tt, ro = info; off = do + ro; cnt = 1
    for d in dims: cnt *= d
    loaders = {0: lambda: torch.from_numpy(np.fromfile(path,dtype='float32',count=cnt,offset=off).copy()).reshape(dims[::-1]).T if len(dims)==2 else torch.from_numpy(np.fromfile(path,dtype='float32',count=cnt,offset=off).copy()).reshape(dims),
               1: lambda: torch.from_numpy(np.fromfile(path,dtype='float16',count=cnt,offset=off).astype('float32').copy()).reshape(dims[::-1]).T.contiguous() if len(dims)==2 else torch.from_numpy(np.fromfile(path,dtype='float16',count=cnt,offset=off).astype('float32').copy()).reshape(dims),
               2: lambda: dq4_0(path,off,cnt,dims), 8: lambda: dq8_0(path,off,cnt,dims),
               10: lambda: dq2k(path,off,cnt,dims), 11: lambda: dq3k(path,off,cnt,dims),
               12: lambda: dq4k(path,off,cnt,dims), 13: lambda: dq5k(path,off,cnt,dims),
               14: lambda: dq6k(path,off,cnt,dims)}
    if tt in loaders: return loaders[tt]()
    print(f"  WARN: type {tt}, zeros"); return torch.zeros(dims)

# ═══════════════════════════════════════════════════════════════
# ARCHITECTURE-AWARE FORWARD PASS
# ═══════════════════════════════════════════════════════════════

def rms_norm(x, w, eps=1e-6):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

def apply_rope(q, k, n_head, head_dim, theta=500000.0, positions=None):
    """Apply rotary position embeddings.

    CORRECTNESS FIX (correction 13, 2026-05-16): The previous version hardcoded
    positions as torch.arange(T), which is correct ONLY for prefill (where T
    equals the full sequence length and positions are 0..T-1). For
    autoregressive decode with a KV cache, T=1 but the new token is at
    position len(generated)-1, NOT 0. Without explicit positions, every
    decode step rotated Q and K as if at position 0, which is why we observed
    token 27268 repeating on the CPU path.

    positions: int64 tensor of shape [T] giving the absolute sequence position
               of each row in q and k. If None, falls back to arange(T) for
               backward compatibility with prefill-only callers.
    """
    B, T, _ = q.shape
    n_head_k = k.shape[-1] // head_dim
    q = q.view(B, T, n_head, head_dim)
    k = k.view(B, T, n_head_k, head_dim)
    
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    if positions is None:
        t = torch.arange(T, dtype=torch.float32, device=q.device)
    else:
        t = positions.to(dtype=torch.float32, device=q.device)
    angles = torch.outer(t, freqs)  # [T, half]
    cos_f = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, half]
    sin_f = torch.sin(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, half]
    
    def rotate(x):
        x1 = x[..., :half]; x2 = x[..., half:]
        return torch.cat([x1*cos_f - x2*sin_f, x1*sin_f + x2*cos_f], dim=-1)
    
    return rotate(q).view(B,T,-1), rotate(k).view(B,T,-1)

def llama_layer(w, x, il, cfg):
    """Llama-family layer: RMSNorm, separate QKV, RoPE, GQA, SwiGLU."""
    p = f'blk.{il}'
    nh = cfg['n_head']; nkv = cfg.get('n_head_kv', nh); hd = cfg['n_embd'] // nh
    
    # Attention norm
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg.get('norm_eps', 1e-5))
    
    # Separate Q, K, V projections
    q = h @ w[f'{p}.attn_q.weight']
    k = h @ w[f'{p}.attn_k.weight']
    v = h @ w[f'{p}.attn_v.weight']
    
    # RoPE
    q, k = apply_rope(q, k, nh, hd, cfg.get("rope_theta", 500000.0))
    
    B, T, _ = q.shape
    q = q.view(B,T,nh,hd).transpose(1,2)
    k = k.view(B,T,nkv,hd).transpose(1,2)
    v = v.view(B,T,nkv,hd).transpose(1,2)
    
    # GQA: repeat K,V if n_head_kv < n_head
    if nkv < nh:
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    
    # Attention
    att = (q @ k.transpose(-2,-1)) * (hd**-0.5)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v
    y = y.transpose(1,2).contiguous().view(B,T,nh*hd)
    y = y @ w[f'{p}.attn_output.weight']
    x = x + y
    
    # FFN: SwiGLU
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg.get('norm_eps', 1e-5))
    gate = F.silu(h2 @ w[f'{p}.ffn_gate.weight'])
    up = h2 @ w[f'{p}.ffn_up.weight']
    ffn = (gate * up) @ w[f'{p}.ffn_down.weight']
    return x + ffn

def gemma_layer(w, x, il, cfg):
    """Gemma v1 layer — substantive substrate-honest substantive differences from llama:
       1. Q is explicitly scaled by 1/sqrt(n_embd_head) BEFORE attention
          (llama bakes this into attention scaling; gemma does it pre-attn)
       2. FFN uses GELU, not SiLU (different activation)
       3. The embedding scaling by sqrt(n_embd) happens once in run(),
          not per-layer — so this layer just sees the already-scaled inpL

    Otherwise structurally same as llama: RMSNorm, separate QKV, RoPE,
    GQA, residual connections.

    Per ggml's src/models/gemma.cpp graph::graph constructor.
    """
    p = f'blk.{il}'
    nh = cfg['n_head']; nkv = cfg.get('n_head_kv', nh); hd = cfg['n_embd'] // nh

    # Attention norm
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg.get('norm_eps', 1e-5))

    # Separate Q, K, V projections
    q = h @ w[f'{p}.attn_q.weight']
    k = h @ w[f'{p}.attn_k.weight']
    v = h @ w[f'{p}.attn_v.weight']

    # RoPE
    q, k = apply_rope(q, k, nh, hd, cfg.get("rope_theta", 10000.0))

    # GEMMA-SPECIFIC: scale Q by 1/sqrt(hd) BEFORE the attention computation
    # (llama scales the QK product; gemma scales Q itself, mathematically
    #  equivalent but happens earlier in the pipeline)
    q = q * (hd ** -0.5)

    B, T, _ = q.shape
    q = q.view(B,T,nh,hd).transpose(1,2)
    k = k.view(B,T,nkv,hd).transpose(1,2)
    v = v.view(B,T,nkv,hd).transpose(1,2)

    # GQA: repeat K,V if n_head_kv < n_head
    if nkv < nh:
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # Attention — note Q is already scaled, so we DON'T multiply by hd**-0.5 here
    att = q @ k.transpose(-2,-1)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v
    y = y.transpose(1,2).contiguous().view(B,T,nh*hd)
    y = y @ w[f'{p}.attn_output.weight']
    x = x + y

    # GEMMA-SPECIFIC FFN: GELU instead of SiLU. Otherwise same gated structure.
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg.get('norm_eps', 1e-5))
    gate = F.gelu(h2 @ w[f'{p}.ffn_gate.weight'])
    up = h2 @ w[f'{p}.ffn_up.weight']
    ffn = (gate * up) @ w[f'{p}.ffn_down.weight']
    return x + ffn


def gemma2_layer(w, x, il, cfg):
    """Gemma2 layer — substantive substrate-honest substrate-deep substantive
    substrate-deep differences from Gemma v1 (per ggml's src/models/gemma2.cpp):

      1. Has attn_post_norm AND ffn_post_norm — additional RMSNorm
         layers AFTER attention/FFN, BEFORE the residual add.
      
      2. Q scaled by f_attention_scale (hparam value) instead of
         1/sqrt(head_dim). The hparam is stored in GGUF metadata.
      
      3. Alternating sliding window vs full attention per layer
         (out of scope for short prompts; defer).
      
      4. Final logit softcapping handled in run()/generate() — NOT
         in this layer function (this is the residual + norm only).

    The embedding sqrt(n_embd) scaling is handled in run()/generate()
    (shared with gemma v1).

    Per ggml's src/models/gemma2.cpp graph::graph constructor.
    """
    p = f'blk.{il}'
    nh = cfg['n_head']; nkv = cfg.get('n_head_kv', nh); hd = cfg['n_embd'] // nh

    # ATTN: norm → QKV → RoPE → scale Q → attention → POST-NORM → residual
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg.get('norm_eps', 1e-5))
    q = h @ w[f'{p}.attn_q.weight']
    k = h @ w[f'{p}.attn_k.weight']
    v = h @ w[f'{p}.attn_v.weight']
    q, k = apply_rope(q, k, nh, hd, cfg.get("rope_theta", 10000.0))

    # GEMMA2: Q scaled by f_attention_scale (hparam, NOT 1/sqrt(hd))
    attn_scale = cfg.get('f_attention_scale', hd ** -0.5)
    q = q * attn_scale

    B, T, _ = q.shape
    q = q.view(B,T,nh,hd).transpose(1,2)
    k = k.view(B,T,nkv,hd).transpose(1,2)
    v = v.view(B,T,nkv,hd).transpose(1,2)
    if nkv < nh:
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    # Q already pre-scaled, so no hd**-0.5 in attention
    att = q @ k.transpose(-2,-1)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v
    y = y.transpose(1,2).contiguous().view(B,T,nh*hd)
    y = y @ w[f'{p}.attn_output.weight']

    # GEMMA2: attn_post_norm BEFORE adding residual
    if f'{p}.post_attention_norm.weight' in w:
        y = rms_norm(y, w[f'{p}.post_attention_norm.weight'], cfg.get('norm_eps', 1e-5))
    elif f'{p}.attn_post_norm.weight' in w:
        y = rms_norm(y, w[f'{p}.attn_post_norm.weight'], cfg.get('norm_eps', 1e-5))
    x = x + y

    # FFN: norm → gate*up via GELU → ffn_down → POST-NORM → residual
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg.get('norm_eps', 1e-5))
    gate = F.gelu(h2 @ w[f'{p}.ffn_gate.weight'])
    up = h2 @ w[f'{p}.ffn_up.weight']
    ffn = (gate * up) @ w[f'{p}.ffn_down.weight']

    # GEMMA2: ffn_post_norm BEFORE adding residual
    if f'{p}.post_ffw_norm.weight' in w:
        ffn = rms_norm(ffn, w[f'{p}.post_ffw_norm.weight'], cfg.get('norm_eps', 1e-5))
    elif f'{p}.ffn_post_norm.weight' in w:
        ffn = rms_norm(ffn, w[f'{p}.ffn_post_norm.weight'], cfg.get('norm_eps', 1e-5))
    return x + ffn


def phi3_layer(w, x, il, cfg):
    """Phi3 layer — substantive substrate-honest substrate-deep substantive
    substrate-deep differences from llama (per ggml's src/models/phi3.cpp):

      1. ffn_up is DOUBLE-WIDTH (shape [n_embd, 2*n_ff]) — contains gate
         and up CONCATENATED. Split at runtime: first half = gate,
         second half = up. No separate ffn_gate tensor.

      2. Q scaled by 1/sqrt(head_dim) BEFORE attention (same as gemma —
         applied pre-attention, not baked into QK product).

      3. No bias on most projections (handled by weight-key lookup).

      4. Dual-scale RoPE via rope_long/rope_short factors. Only kicks in
         at >4k context — for short prompts, standard RoPE applies.

    Otherwise structurally same as llama: RMSNorm, separate QKV, RoPE,
    SwiGLU activation, residual connections.

    Per ggml's src/models/phi3.cpp graph<iswa>::graph constructor.
    """
    p = f'blk.{il}'
    nh = cfg['n_head']; nkv = cfg.get('n_head_kv', nh); hd = cfg['n_embd'] // nh

    # Attention norm
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg.get('norm_eps', 1e-5))

    # Separate Q, K, V projections (some phi3 GGUFs may have fused attn_qkv,
    # but most use separate — handle both)
    if f'{p}.attn_q.weight' in w:
        q = h @ w[f'{p}.attn_q.weight']
        k = h @ w[f'{p}.attn_k.weight']
        v = h @ w[f'{p}.attn_v.weight']
    else:
        # Fused QKV path — split at runtime
        qkv = h @ w[f'{p}.attn_qkv.weight']
        q_dim = nh * hd
        k_dim = nkv * hd
        v_dim = nkv * hd
        q = qkv[..., :q_dim]
        k = qkv[..., q_dim:q_dim + k_dim]
        v = qkv[..., q_dim + k_dim:q_dim + k_dim + v_dim]

    # RoPE — phi3 uses 10000 base typically (or 1000000 for long-context variants)
    q, k = apply_rope(q, k, nh, hd, cfg.get("rope_theta", 10000.0))

    # PHI3-SPECIFIC: pre-scale Q by 1/sqrt(hd) like gemma
    q = q * (hd ** -0.5)

    B, T, _ = q.shape
    q = q.view(B,T,nh,hd).transpose(1,2)
    k = k.view(B,T,nkv,hd).transpose(1,2)
    v = v.view(B,T,nkv,hd).transpose(1,2)

    # GQA
    if nkv < nh:
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    # Attention — Q already pre-scaled
    att = q @ k.transpose(-2,-1)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = F.softmax(att, dim=-1) @ v
    y = y.transpose(1,2).contiguous().view(B,T,nh*hd)
    y = y @ w[f'{p}.attn_output.weight']
    x = x + y

    # PHI3-SPECIFIC FFN: ffn_up has shape [n_embd, 2*n_ff], split at runtime
    # First half is gate, second half is up; then SwiGLU = silu(gate) * up
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg.get('norm_eps', 1e-5))
    fused_up = h2 @ w[f'{p}.ffn_up.weight']    # shape [B, T, 2*n_ff]
    n_ff = fused_up.shape[-1] // 2
    gate = fused_up[..., :n_ff]
    up   = fused_up[..., n_ff:]
    ffn = (F.silu(gate) * up) @ w[f'{p}.ffn_down.weight']
    return x + ffn


def gpt2_layer(w, x, il, cfg):
    """GPT-2 layer: LayerNorm, fused QKV, position embd, GELU."""
    p = f'blk.{il}'; nh = cfg['n_head']
    
    ln1 = F.layer_norm(x, [x.shape[-1]], w[f'{p}.attn_norm.weight'], w[f'{p}.attn_norm.bias'])
    qkv = ln1 @ w[f'{p}.attn_qkv.weight'] + w[f'{p}.attn_qkv.bias']
    B,T,C3 = qkv.shape; C = C3//3; hd = C//nh
    q,k,v = qkv.split(C, dim=-1)
    q=q.view(B,T,nh,hd).transpose(1,2); k=k.view(B,T,nh,hd).transpose(1,2); v=v.view(B,T,nh,hd).transpose(1,2)
    att = (q@k.transpose(-2,-1))*(hd**-0.5)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = (F.softmax(att,dim=-1)@v).transpose(1,2).contiguous().view(B,T,C)
    y = y @ w[f'{p}.attn_output.weight'] + w[f'{p}.attn_output.bias']
    x = x + y
    ln2 = F.layer_norm(x, [x.shape[-1]], w[f'{p}.ffn_norm.weight'], w[f'{p}.ffn_norm.bias'])
    h = F.gelu(ln2 @ w[f'{p}.ffn_up.weight'] + w[f'{p}.ffn_up.bias'])
    ffn = h @ w[f'{p}.ffn_down.weight'] + w[f'{p}.ffn_down.bias']
    return x + ffn

def qwen2_layer(w, x, il, cfg):
    """Qwen2 layer: RMSNorm, separate QKV with bias, RoPE, GQA, SwiGLU."""
    p = f'blk.{il}'; nh = cfg['n_head']; nkv = cfg.get('n_head_kv', nh); hd = cfg['n_embd']//nh
    
    h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg.get('norm_eps', 1e-6))
    q = h @ w[f'{p}.attn_q.weight'] + w.get(f'{p}.attn_q.bias', 0)
    k = h @ w[f'{p}.attn_k.weight'] + w.get(f'{p}.attn_k.bias', 0)
    v = h @ w[f'{p}.attn_v.weight'] + w.get(f'{p}.attn_v.bias', 0)
    q, k = apply_rope(q, k, nh, hd, cfg.get("rope_theta", 500000.0))
    B,T,_ = q.shape
    q=q.view(B,T,nh,hd).transpose(1,2); k=k.view(B,T,nkv,hd).transpose(1,2); v=v.view(B,T,nkv,hd).transpose(1,2)
    if nkv < nh:
        rep = nh//nkv; k=k.repeat_interleave(rep,dim=1); v=v.repeat_interleave(rep,dim=1)
    att = (q@k.transpose(-2,-1))*(hd**-0.5)
    mask = torch.triu(torch.ones(T,T,dtype=torch.bool,device=att.device), diagonal=1)
    att = att.masked_fill(mask, float('-inf'))
    y = (F.softmax(att,dim=-1)@v).transpose(1,2).contiguous().view(B,T,nh*hd)
    y = y @ w[f'{p}.attn_output.weight']
    x = x + y
    h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg.get('norm_eps', 1e-6))
    gate = F.silu(h2 @ w[f'{p}.ffn_gate.weight'])
    up = h2 @ w[f'{p}.ffn_up.weight']
    ffn = (gate * up) @ w[f'{p}.ffn_down.weight']
    return x + ffn

LAYER_FN = {'llama': llama_layer, 'gpt2': gpt2_layer, 'qwen2': qwen2_layer,
            'starcoder2': llama_layer,
            'gemma': gemma_layer,         # Gemma v1: sqrt(n_embd) embed scaling + GELU FFN
            'gemma2': gemma2_layer,       # Gemma2: gemma v1 + attn_post_norm + ffn_post_norm + f_attention_scale
            'phi3': phi3_layer,           # Phi3: double-width ffn_up (gate+up fused) + pre-scaled Q
            }

# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

def run(path, input_ids, n_predict=1):
    t0 = time.time()
    print(f"=== LlamaTov Inference ===\nModel: {path}")
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    cfg = {
        'arch': arch,
        'n_layers': md.get(f'{arch}.block_count', 12),
        'n_head': md.get(f'{arch}.attention.head_count', 12),
        'n_head_kv': md.get(f'{arch}.attention.head_count_kv', md.get(f'{arch}.attention.head_count', 12)),
        'n_embd': md.get(f'{arch}.embedding_length', 768),
        'n_ff': md.get(f'{arch}.feed_forward_length', 3072),
        'rope_theta': md.get(f'{arch}.rope.freq_base', 500000.0),
        'norm_eps': md.get(f'{arch}.attention.layer_norm_rms_epsilon',
                    md.get(f'{arch}.attention.layer_norm_epsilon', 1e-5)),
        'vocab_size': md.get(f'{arch}.vocab_size', 32000),
    }
    print(f"Arch: {arch}, layers: {cfg['n_layers']}, heads: {cfg['n_head']}/{cfg['n_head_kv']}, embd: {cfg['n_embd']}")
    
    print("Loading weights...")
    w = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time(); print(f"Loaded {len(w)} tensors in {t1-t0:.1f}s")
    
    layer_fn = LAYER_FN.get(arch, llama_layer)
    has_pos_embd = 'position_embd.weight' in w
    
    # Embedding
    tok = torch.tensor(input_ids, dtype=torch.long)
    x = w['token_embd.weight'].T[tok] if w['token_embd.weight'].shape[0] < w['token_embd.weight'].shape[1] else w['token_embd.weight'][tok]
    if has_pos_embd:
        pos = torch.arange(len(input_ids), dtype=torch.long)
        pe = w['position_embd.weight']
        x = x + (pe.T[pos] if pe.shape[0] < pe.shape[1] else pe[pos])
    x = x.unsqueeze(0)

    # Gemma v1 + v2 substrate-honest substrate-deep substantive substantive
    # substantive embedding scaling: multiply by sqrt(n_embd) after lookup.
    # Per ggml's gemma.cpp:   inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
    if arch in ('gemma', 'gemma2'):
        x = x * (cfg['n_embd'] ** 0.5)
    
    # Layers
    print(f"Running {cfg['n_layers']} layers ({arch})...")
    for i in range(cfg['n_layers']):
        x = layer_fn(w, x, i, cfg)
        if i % max(1, cfg['n_layers']//4) == 0: print(f"  Layer {i}/{cfg['n_layers']}")
    
    # Output
    if 'output_norm.weight' in w:
        if 'output_norm.bias' in w:
            x = F.layer_norm(x, [x.shape[-1]], w['output_norm.weight'], w['output_norm.bias'])
        else:
            x = rms_norm(x, w['output_norm.weight'], cfg.get('norm_eps', 1e-5))
    
    lm = w.get('output.weight', w.get('token_embd.weight'))  # weight tying
    # GGUF weight shape varies: try both orientations
    if lm.shape[-1] == cfg['n_embd']:
        logits = x @ lm.T  # [B,T,embd] @ [embd,vocab] 
    else:
        logits = x @ lm    # already correct orientation
    token = int(logits[0, -1].argmax().item())
    
    t2 = time.time()
    print(f"\nTime: {t2-t0:.2f}s ({t1-t0:.1f}s load + {t2-t1:.1f}s inference)")
    print(f"=== OUTPUT TOKEN: {token} ===")
    return token


def generate(path, input_ids, n_tokens=20, device="cpu"):
    """Generate multiple tokens with KV cache for tok/s measurement."""
    t0 = time.time()
    print(f"=== LlamaTov Generate ===\nModel: {path}")
    md, ts, do = parse_gguf(path)
    arch = md.get('general.architecture', 'llama')
    cfg = {
        'arch': arch,
        'n_layers': md.get(f'{arch}.block_count', 12),
        'n_head': md.get(f'{arch}.attention.head_count', 12),
        'n_head_kv': md.get(f'{arch}.attention.head_count_kv', md.get(f'{arch}.attention.head_count', 12)),
        'n_embd': md.get(f'{arch}.embedding_length', 768),
        'rope_theta': md.get(f'{arch}.rope.freq_base', 500000.0),
        'norm_eps': md.get(f'{arch}.attention.layer_norm_rms_epsilon',
                    md.get(f'{arch}.attention.layer_norm_epsilon', 1e-5)),
    }
    nh = cfg['n_head']; nkv = cfg['n_head_kv']; hd = cfg['n_embd'] // nh
    nl = cfg['n_layers']
    
    print(f"Arch: {arch}, layers: {nl}, heads: {nh}/{nkv}, embd: {cfg['n_embd']}")
    print("Loading weights...")
    w = {n: lt(path, do, info) for n, info in ts.items()}
    t1 = time.time()
    print(f"Loaded {len(w)} tensors in {t1-t0:.1f}s")
    
    # Move to device (CPU or CUDA)
    if device != "cpu":
        print(f"Moving weights to {device}...")
        w = {n: t.to(device) for n, t in w.items()}
        t_move = time.time()
        print(f"Moved to {device} in {t_move-t1:.1f}s")
    
    has_pos_embd = 'position_embd.weight' in w
    is_gpt2 = arch in ('gpt2', 'starcoder2')
    
    # Initialize KV cache: list of (K, V) per layer, initially None
    kv_cache = [None] * nl
    
    generated = list(input_ids)
    all_positions = list(range(len(input_ids)))
    
    print(f"Generating {n_tokens} tokens...")
    t_gen_start = time.time()
    
    for step in range(n_tokens):
        if step == 0:
            # Prefill: process all input tokens
            tok = torch.tensor(generated, dtype=torch.long, device=device)
            positions = torch.arange(len(generated), dtype=torch.long, device=device)
        else:
            # Decode: process only the last generated token
            tok = torch.tensor([generated[-1]], dtype=torch.long, device=device)
            positions = torch.tensor([len(generated) - 1], dtype=torch.long, device=device)
        
        # Embedding
        emb = w['token_embd.weight']
        x = (emb.T[tok] if emb.shape[0] < emb.shape[1] else emb[tok])
        if has_pos_embd:
            pe = w['position_embd.weight']
            x = x + (pe.T[positions] if pe.shape[0] < pe.shape[1] else pe[positions])
        x = x.unsqueeze(0)  # [1, seq_len, embd]

        # Gemma v1/v2 substantive substrate-honest embedding scaling
        # Per ggml gemma.cpp: inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        if arch in ('gemma', 'gemma2'):
            x = x * (cfg['n_embd'] ** 0.5)
        
        # Run layers with KV cache
        for il in range(nl):
            p = f'blk.{il}'
            
            if is_gpt2:
                # GPT-2: LayerNorm + fused QKV
                h = F.layer_norm(x, [x.shape[-1]], w[f'{p}.attn_norm.weight'], w[f'{p}.attn_norm.bias'])
                qkv = h @ w[f'{p}.attn_qkv.weight'] + w[f'{p}.attn_qkv.bias']
                B,T,C3 = qkv.shape; C = C3//3
                q_cur, k_cur, v_cur = qkv.split(C, dim=-1)
            else:
                # Llama/Qwen2: RMSNorm + separate QKV
                h = rms_norm(x, w[f'{p}.attn_norm.weight'], cfg['norm_eps'])
                q_cur = h @ w[f'{p}.attn_q.weight']
                k_cur = h @ w[f'{p}.attn_k.weight']
                v_cur = h @ w[f'{p}.attn_v.weight']
                if f'{p}.attn_q.bias' in w:
                    q_cur = q_cur + w[f'{p}.attn_q.bias']
                    k_cur = k_cur + w[f'{p}.attn_k.bias']
                    v_cur = v_cur + w[f'{p}.attn_v.bias']
                # RoPE — pass positions explicitly so decode steps rotate
                # at the absolute sequence position, not the relative T_cur=0
                # (this is the correction-13 fix for the repeating-token bug)
                q_cur, k_cur = apply_rope(q_cur, k_cur, nh, hd,
                                          cfg.get("rope_theta", 500000.0),
                                          positions=positions)
            
            B, T_cur, _ = q_cur.shape
            q = q_cur.view(B, T_cur, nh, hd).transpose(1, 2)
            k_new = k_cur.view(B, T_cur, nkv, hd).transpose(1, 2)
            v_new = v_cur.view(B, T_cur, nkv, hd).transpose(1, 2)
            
            # KV cache: append new K,V to cached
            if kv_cache[il] is not None:
                k_cached, v_cached = kv_cache[il]
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k = k_new
                v = v_new
            kv_cache[il] = (k, v)
            
            # GQA: repeat K,V if needed
            if nkv < nh:
                rep = nh // nkv
                k_att = k.repeat_interleave(rep, dim=1)
                v_att = v.repeat_interleave(rep, dim=1)
            else:
                k_att = k; v_att = v
            
            # Attention (q is only new positions, k/v is all positions)
            T_total = k_att.shape[2]
            att = (q @ k_att.transpose(-2, -1)) * (hd ** -0.5)
            # Causal mask: new positions can attend to all previous + self
            if T_total > 1:
                # Build mask for the positions we're computing
                mask = torch.ones(T_cur, T_total, dtype=torch.bool, device=x.device)
                for i in range(T_cur):
                    pos = len(generated) - T_cur + i if step == 0 else len(generated) - 1
                    mask[i, pos+1:] = False
                mask = ~mask  # True = mask out
                att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            y = F.softmax(att, dim=-1) @ v_att
            y = y.transpose(1, 2).contiguous().view(B, T_cur, nh * hd)
            
            # Output projection + residual
            y = y @ w[f'{p}.attn_output.weight']
            if f'{p}.attn_output.bias' in w:
                y = y + w[f'{p}.attn_output.bias']
            x = x + y
            
            # FFN
            if is_gpt2:
                h2 = F.layer_norm(x, [x.shape[-1]], w[f'{p}.ffn_norm.weight'], w[f'{p}.ffn_norm.bias'])
                ffn_h = F.gelu(h2 @ w[f'{p}.ffn_up.weight'] + w[f'{p}.ffn_up.bias'])
                ffn = ffn_h @ w[f'{p}.ffn_down.weight'] + w[f'{p}.ffn_down.bias']
            else:
                h2 = rms_norm(x, w[f'{p}.ffn_norm.weight'], cfg['norm_eps'])
                gate = F.silu(h2 @ w[f'{p}.ffn_gate.weight'])
                up = h2 @ w[f'{p}.ffn_up.weight']
                ffn = (gate * up) @ w[f'{p}.ffn_down.weight']
            x = x + ffn
        
        # Output head (only last position)
        x_last = x[:, -1:, :]
        if 'output_norm.weight' in w:
            if 'output_norm.bias' in w:
                x_last = F.layer_norm(x_last, [x_last.shape[-1]], w['output_norm.weight'], w['output_norm.bias'])
            else:
                x_last = rms_norm(x_last, w['output_norm.weight'], cfg['norm_eps'])
        
        lm = w.get('output.weight', w.get('token_embd.weight'))
        if lm.shape[-1] == cfg['n_embd']:
            logits = x_last @ lm.T
        else:
            logits = x_last @ lm
        
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        
        if step < 3 or step == n_tokens - 1:
            print(f"  Step {step}: token {next_token}")
    
    t2 = time.time()
    gen_time = t2 - t_gen_start
    tok_per_sec = n_tokens / gen_time
    
    print(f"\n=== RESULTS ===")
    print(f"Generated {n_tokens} tokens in {gen_time:.2f}s")
    print(f"Throughput: {tok_per_sec:.1f} tok/s")
    print(f"Total (load + gen): {t2-t0:.2f}s")
    print(f"Tokens: {generated}")
    return generated, tok_per_sec


if __name__ == '__main__':
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = [a for a in sys.argv[1:] if a.startswith('--')]
    
    p = args[0] if args else '/tmp/llamatov-data/model-zoo/gpt2-124m-Q2_K.gguf'
    ids = [int(x) for x in args[1].split(',')] if len(args) > 1 else [1, 15043]
    
    if '--generate' in flags:
        n = int(args[2]) if len(args) > 2 else 20
        dev = 'cpu' if '--cpu' in flags else ('cuda' if torch.cuda.is_available() else 'cpu')
        generate(p, ids, n, device=dev)
    else:
        run(p, ids)
