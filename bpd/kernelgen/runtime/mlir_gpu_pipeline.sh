#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# mlir_gpu_pipeline.sh OP  — lower a generated <OP>_gpu_gen.mlir (plain-pointer,
# 3-param kernel) to PTX (elementwise) or libdevice-linked cubin (transcendental),
# inject the param-doc header, and report. Companion launchers: cubin_launch (cubin),
# named_launch (PTX). Writes <OP>_mlirgpu.{ptx,cubin}.
#
# Pipeline mechanics (the proven path):
#   --convert-arith-to-llvm        (the arith body ops in the llvm.func)
#   --nvvm-attach-target chip=sm_61 [l=libdevice.10.bc for transcendentals]
#   --convert-gpu-to-nvvm --reconcile-unrealized-casts
#   --gpu-module-to-binary format={isa|bin}
# Transcendentals (math.tanh/erf/exp -> __nv_*) MUST link libdevice (l=...) and
# use format=bin (ptxas leaves __nv_* extern otherwise). Elementwise -> isa/PTX.
set -e
OP="$1"
MB=/tmp/gpu-work/mlir_backend
CUDA=/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8
LD="$CUDA/nvvm/libdevice/libdevice.10.bc"
export PATH=/run/current-system/sw/bin:$PATH
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$CUDA/lib
cd "$MB"
SRC="${OP}_gpu_gen.mlir"

# Does this kernel use transcendentals (math.* -> libdevice)?
if grep -qE "math\.(tanh|erf|exp|log|sin|cos)" "$SRC"; then
  NEEDS_LD=1; TGT="chip=sm_61 O=3 l=$LD"; FMT="bin"; EXT="cubin"
else
  NEEDS_LD=0; TGT="chip=sm_61 O=3"; FMT="isa"; EXT="ptx"
fi

mlir-opt "$SRC" \
  --convert-arith-to-llvm \
  --nvvm-attach-target="$TGT" \
  --convert-gpu-to-nvvm \
  --reconcile-unrealized-casts \
  --gpu-module-to-binary="format=$FMT" 2>/dev/null > "${OP}_gpu_lowered.out"

# extract the binary (assembly= for PTX text, bin= for cubin bytes) + inject param doc.
python3 - "$OP" "$FMT" "$EXT" << 'PYEOF'
import re, sys
op, fmt, ext = sys.argv[1], sys.argv[2], sys.argv[3]
out = open(f"{op}_gpu_lowered.out").read()
# the // param-doc header lines from the source .mlir
hdr = [l for l in open(f"{op}_gpu_gen.mlir").read().splitlines() if l.startswith("//")]
# ASCII-sanitize: ptxas JIT (cuModuleLoadData) is ASCII-strict; em-dash etc. break it.
banner = ("\n".join(hdr) + "\n//\n").encode("ascii", "replace").decode("ascii")
if fmt == "isa":
    m = re.search(r'assembly = "(.*?)">\]', out, re.DOTALL)
    ptx = (m.group(1).replace("\\0A","\n").replace("\\09","\t")
           .replace("\\00","").replace('\\22','"').replace("\\5C","\\"))
    # inject the param doc after NVPTX's own leading // block
    lines = ptx.splitlines(keepends=True); i = 0
    while i < len(lines) and (lines[i].startswith("//") or lines[i].strip()==""):
        i += 1
    open(f"{op}_mlirgpu.ptx","w").write("".join(lines[:i]) + banner + "".join(lines[i:]))
    print(f"  {op}: PTX (elementwise, {len(ptx)}B) + param-doc injected -> {op}_mlirgpu.ptx")
else:
    m = re.search(r'bin = "(.*?)">\]', out, re.DOTALL)
    raw = m.group(1); b = bytearray(); k = 0
    while k < len(raw):
        c = raw[k]
        if c == "\\":
            n = raw[k+1]
            if n == "\\": b.append(0x5C); k += 2
            elif n == '"': b.append(0x22); k += 2
            else: b.append(int(raw[k+1:k+3],16)); k += 3
        else: b.append(ord(c)); k += 1
    open(f"{op}_mlirgpu.cubin","wb").write(b)
    # cubin is binary; write the param doc to a sidecar .params
    open(f"{op}_mlirgpu.params","w").write(banner)
    print(f"  {op}: cubin (transcendental+libdevice, {len(b)}B) -> {op}_mlirgpu.cubin (+ .params doc)")
PYEOF

echo "PIPELINE_DONE ${OP} format=${FMT} libdevice=${NEEDS_LD}"
