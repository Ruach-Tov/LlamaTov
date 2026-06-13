# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""Count subnormal F16 scales in blk.0.attn_q.weight (the actual count, fixed numpy stride)."""
import sys
sys.path.insert(0, "/tmp/bpd_test/bench")
import numpy as np
from gguf_helper import query_tensor, read_tensor_bytes

gguf = "/tmp/llamatov-data/ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
info = query_tensor(gguf, "blk.0.attn_q.weight")
raw = read_tensor_bytes(gguf, info)
n_blocks = len(raw) // 34
print(f"attn_q.weight: {n_blocks} Q8_0 blocks")

# Properly extract scale bytes: bytes 0 and 1 of every 34-byte block
# Reshape raw as (n_blocks, 34), then view bytes [0:2] as uint16
raw_np = np.frombuffer(raw, dtype=np.uint8)
blocks_2d = raw_np.reshape(n_blocks, 34)
scales_u16 = blocks_2d[:, :2].copy().view(np.uint16).reshape(n_blocks)

exp_bits = (scales_u16 >> 10) & 0x1F
n_subnormal = int((exp_bits == 0).sum())
n_normal = int(((exp_bits > 0) & (exp_bits < 31)).sum())
n_inf_nan = int((exp_bits == 31).sum())
print(f"subnormal: {n_subnormal}, normal: {n_normal}, inf/nan: {n_inf_nan}")

if n_subnormal > 0:
    # Print first few subnormal blocks
    subnormal_idxs = np.where(exp_bits == 0)[0][:5]
    for i in subnormal_idxs:
        print(f"  Block {i}: scale_u16 = {scales_u16[i]:#06x}")
