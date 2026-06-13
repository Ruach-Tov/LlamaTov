#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# build_eval_callback.sh — build a patched llama.cpp eval-callback that dumps
# binary tensors when LLAMA_DUMP_DIR is set.
#
# Output: <llama_cpp_dir>/build/bin/llama-eval-callback
#
# This is the substrate-design-canonical reference build for the LlamaTov
# correctness harness. Anyone running the harness should build their own
# copy of this tool with this exact patch, ensuring fixtures are
# byte-comparable across verifiers.

set -e

if [ -z "$1" ]; then
    echo "usage: $0 <path-to-llama.cpp-source>"
    echo ""
    echo "  Expects llama.cpp source at the given path. The script will:"
    echo "    1. Apply our binary-dump patch to examples/eval-callback/eval-callback.cpp"
    echo "    2. Configure cmake (CPU-only, no CUDA, no BLAS, no CURL)"
    echo "    3. Build llama-eval-callback target"
    echo ""
    echo "  The resulting binary will dump binary tensor snapshots to"
    echo "  LLAMA_DUMP_DIR for each ggml operation in the inference graph."
    exit 2
fi

LLAMA_CPP_DIR="$1"
PATCHER="$(dirname "$0")/../../bench/apply_eval_callback_patch.py"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "error: $LLAMA_CPP_DIR does not exist"
    exit 1
fi
if [ ! -f "$PATCHER" ]; then
    echo "error: patcher not found at $PATCHER"
    exit 1
fi

cd "$LLAMA_CPP_DIR"
echo "[build] applying binary-dump patch to eval-callback.cpp"
# The patcher hardcodes /tmp/llama_cpp_test as the path; override via env or symlink.
# For portability we'll temporarily symlink the user's llama.cpp into the expected path.
PATCH_TARGET="$LLAMA_CPP_DIR/examples/eval-callback/eval-callback.cpp"
if [ ! -f "$PATCH_TARGET" ]; then
    echo "error: $PATCH_TARGET not found (is this really llama.cpp source?)"
    exit 1
fi

# Modify the patcher to accept a path argument, OR copy the source to where it expects.
# Simplest: edit the patcher's hardcoded path inline.
python3 - <<PYEOF
import re
with open("$PATCHER") as f:
    src = f.read()
src = src.replace('/tmp/llama_cpp_test/examples/eval-callback/eval-callback.cpp',
                   '$PATCH_TARGET')
with open('/tmp/_patcher_local.py', 'w') as f:
    f.write(src)
PYEOF
python3 /tmp/_patcher_local.py

echo "[build] configuring cmake (CPU only)"
mkdir -p build && cd build
cmake .. \
  -DGGML_CUDA=OFF \
  -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

echo "[build] building llama-eval-callback target"
make -j$(nproc) llama-eval-callback

echo ""
echo "[done] built: $LLAMA_CPP_DIR/build/bin/llama-eval-callback"
echo ""
echo "  usage: LLAMA_DUMP_DIR=/tmp/my_fixture \\"
echo "         $LLAMA_CPP_DIR/build/bin/llama-eval-callback \\"
echo "         -m /path/to/llama3.2-1b.gguf \\"
echo "         -p 'Hello, my name is' -n 1 --temp 0 --seed 42 -c 64 -t 2"
