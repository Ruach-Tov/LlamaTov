# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
cd ~/Ruach-Tov
echo "=== all-ON (production) ==="
BPD_BIAS_FOLD=1 BPD_APPEND_KV_FUSED=1 BPD_APPEND_INCR_FUSED=1 python3 -u /tmp/cold_iso.py 2>&1 | grep -E "argmax|Error" | tail -1
echo "=== BIAS_FOLD=0 alone ==="
BPD_BIAS_FOLD=0 BPD_APPEND_KV_FUSED=1 BPD_APPEND_INCR_FUSED=1 python3 -u /tmp/cold_iso.py 2>&1 | grep -E "argmax|Error" | tail -1
echo "=== APPEND_KV=0 alone ==="
BPD_BIAS_FOLD=1 BPD_APPEND_KV_FUSED=0 BPD_APPEND_INCR_FUSED=1 python3 -u /tmp/cold_iso.py 2>&1 | grep -E "argmax|Error" | tail -1
echo "=== APPEND_INCR=0 alone ==="
BPD_BIAS_FOLD=1 BPD_APPEND_KV_FUSED=1 BPD_APPEND_INCR_FUSED=0 python3 -u /tmp/cold_iso.py 2>&1 | grep -E "argmax|Error" | tail -1
echo "=== all-OFF (Bocher's OFF arm) ==="
BPD_BIAS_FOLD=0 BPD_APPEND_KV_FUSED=0 BPD_APPEND_INCR_FUSED=0 python3 -u /tmp/cold_iso.py 2>&1 | grep -E "argmax|Error" | tail -1
