# SPDX-License-Identifier: LicenseRef-RTAAL-1.1
#!/usr/bin/env python3
"""verify_dq5_0.py — verify our dq5_0 against an INDEPENDENT ggml-spec numpy reference, on the real
Ollama Q5_0 blocks. This is the correctness gate (not 'values look sane' but 'matches the spec')."""
import sys, struct
sys.path.insert(0, "os.environ.get("BPD","bpd")"); sys.path.insert(0, "os.environ.get("BPD","bpd")/lib")
import numpy as np, llamatov_run as R

BLOB = "models/qwen2.5-0.5b.gguf"

def ggml_ref_q5_0(raw_bytes, nb):
    """Independent textbook ggml dequantize_row_q5_0 — scalar loop, follows the C source exactly.
    block: d(fp16,2) qh(uint32,4) qs(16). For i in 0..15:
      xh_0 = ((qh >> (i+0))  << 4) & 0x10 ; xh_1 = ((qh >> (i+12)) >> 0) & 0x10  (per ggml)
      x[i]    = d * ((qs[i]&0xF) | xh_0) - 16
      x[i+16] = d * ((qs[i]>>4)  | xh_1) - 16
    (ggml ggml-quants.c dequantize_row_q5_0)"""
    out = np.empty(nb*32, np.float32)
    for b in range(nb):
        base = b*22
        d = np.frombuffer(raw_bytes[base:base+2], dtype=np.float16)[0].astype(np.float32)
        qh = struct.unpack('<I', raw_bytes[base+2:base+6])[0]
        qs = raw_bytes[base+6:base+22]
        for i in range(16):
            xh_0 = ((qh >> (i + 0)) << 4) & 0x10
            xh_1 = ((qh >> (i + 12))) & 0x10
            x0 = ((qs[i] & 0x0F) | xh_0) - 16
            x1 = ((qs[i] >> 4)   | xh_1) - 16
            out[b*32 + i]      = d * x0
            out[b*32 + i + 16] = d * x1
    return out

def main():
    md, ts, do = R.parse_gguf(BLOB)
    name = "blk.0.attn_k.weight"
    off, qtype, _ = ts[name]  # (shape, type, byteoffset)? confirm layout
    info = ts[name]
    print("tensor info:", info)
    # our dq5_0 result (flattened, pre-transpose comparison is cleanest on raw block order)
    ours = np.asarray(R.lt(BLOB, do, ts[name])).T.reshape(-1)  # undo the .T to get ggml block order
    nb = ours.shape[0] // 32
    # read the raw bytes for the reference
    shape, tt, byteoff = info
    with open(BLOB, 'rb') as f:
        f.seek(do + byteoff); raw = f.read(nb*22)
    ref = ggml_ref_q5_0(raw, nb)
    n = min(len(ours), len(ref))
    diff = np.abs(ours[:n] - ref[:n]).max()
    print(f"  our dq5_0 vs ggml-spec reference: max|diff| = {diff:.6g} over {n} elements")
    print(f"  {'PASS — dq5_0 matches ggml spec' if diff < 1e-5 else 'FAIL — dq5_0 packing is WRONG'}")
    print(f"  sample ours[:5]={ours[:5]}")
    print(f"  sample ref [:5]={ref[:5]}")

if __name__ == "__main__":
    main()
