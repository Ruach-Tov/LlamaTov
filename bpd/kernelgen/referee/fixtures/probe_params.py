# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
import os, re
KB="/tmp/llamatov-data/kernelbench/KernelBench/level1"
# for a conv + a pool + a norm, show __init__ signature + get_init_inputs return
for fn in ["50_conv_standard_2D__square_input__square_kernel","41_Max_Pooling_1D","33_BatchNorm","40_LayerNorm","35_GroupNorm_"]:
    cand=[x for x in os.listdir(KB) if x.startswith(fn.split("_")[0]+"_")]
    if not cand: continue
    src=open(os.path.join(KB,cand[0])).read()
    init=re.search(r"def __init__\(self,([^\)]*)\)", src, re.DOTALL)
    layer=re.search(r"(nn\.\w+\([^\)]*\))", src)
    gii=re.search(r"def get_init_inputs.*?return (\[[^\]]*\])", src, re.DOTALL)
    print(f"=== {cand[0]} ===")
    print(f"  init: {init.group(1).strip()[:75] if init else '?'}")
    print(f"  layer: {layer.group(1)[:60] if layer else '?'}")
    print(f"  get_init_inputs: {gii.group(1).strip()[:55] if gii else '?'}")
