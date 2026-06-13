#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
cd "$(dirname "$0")"/../../bpd
echo "=== 1. read the PRODUCTION gguf: counts must match ggufq (290 tensors, 34 KVs) ==="
swipl -q -g 'consult("lib/gguf_native_reader.pl"), gguf_read_full("models/qwen_q8.gguf", Version, Metadata, Tensors, _), length(Metadata, NM), length(Tensors, NT), format("version=~w metadata=~w tensors=~w~n", [Version, NM, NT]), halt' 2>/dev/null
echo "=== 2. failed-test injection: truncated file (cut mid-header) must fail LOUD not garbage ==="
head -c 16 models/qwen_q8.gguf > /tmp/trunc16.gguf
swipl -q -g 'consult("lib/gguf_native_reader.pl"), catch( (gguf_read_full("/tmp/trunc16.gguf", V, _, _, _), format("READ-OK?! version=~w -> READER-BROKEN (accepted truncated header)~n",[V])), E, (format("refused: ~w -> READER-OK~n",[E]))), halt' 2>/dev/null
echo "=== 3. wrong magic must refuse ==="
printf 'XXXX' > /tmp/badmagic.gguf; head -c 100 models/qwen_q8.gguf | tail -c 96 >> /tmp/badmagic.gguf
swipl -q -g 'consult("lib/gguf_native_reader.pl"), catch( (gguf_read_full("/tmp/badmagic.gguf", _, _, _, _), format("READ-OK?! -> READER-BROKEN (accepted bad magic)~n")), E, format("refused: ~w -> READER-OK~n",[E])), halt' 2>/dev/null
echo "=== 4. string-length-overflow attack fixture (claims huge string) ==="
swipl -q -g 'consult("lib/gguf_native_reader.pl"), catch( (gguf_read_full("tests/crossword_attacks/string_length_overflow.gguf", _, _, _, _), format("READ-OK?! -> check safe_read guards~n")), E, format("refused: ~w -> READER-OK~n",[E])), halt' 2>/dev/null
