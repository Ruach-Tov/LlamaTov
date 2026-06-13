#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
cd "$(dirname "$0")"/../../bpd
for f in tests/crossword_attacks/*.gguf; do
  name=$(basename "$f" .gguf)
  out=$(swipl -q -g "consult(\"lib/gguf_validate.pl\"), gguf_validate(\"$f\"), halt" 2>/dev/null | grep -E "FAIL|PASS|ERROR|reject|valid" | head -3 | tr "\n" " ")
  echo "$name: $out"
done
echo "=== control: the PRODUCTION gguf must PASS ==="
swipl -q -g "consult(\"lib/gguf_validate.pl\"), gguf_validate(\"models/qwen_q8.gguf\"), halt" 2>/dev/null | grep -E "FAIL|PASS|ERROR" | head -4
