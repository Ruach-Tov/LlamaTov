#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# check_soa_layout.sh — detect the SoA-repack layout-mismatch bug class at build time.
# Flags: a function/file that computes SoA offsets (X*32 stride) BUT still uses the
# block_q8_0 STRUCT (34-byte) layout (->d, ->qs, (block_q8_0*)+kbx). That contradiction
# = silent wrong output (reads garbage, no crash). Catches it BEFORE the token gate.
f="$1"
soa=$(grep -cE "kbx[a-z_]* *\* *32|bpr *\* *32|row_base \+ kbx" "$f")
struct=$(grep -nE "\(const block_q8_0 \*\) *[a-z_]+ \+ kbx|bq8_0->(d|qs)" "$f")
if [ "$soa" -gt 0 ] && [ -n "$struct" ]; then
  echo "LAYOUT-MISMATCH in $f: SoA offset math present, but block_q8_0 struct-stride still used:"
  echo "$struct" | sed "s/^/  /"
  echo "  -> the struct (34B) contradicts the SoA repack (32B stride) = silent wrong output. Fix the struct accesses."
  exit 1
fi
echo "OK: $f — no SoA/struct layout contradiction"
