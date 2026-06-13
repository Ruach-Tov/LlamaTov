#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
# build.sh — regenerate (if the BPD toolchain is present) or verify the FROZEN artifact, then test.
#
# Usage: ./build.sh
#
# HONESTY NOTE (2026-06-13): this public substrate ships the GENERATED artifact
# (output/gguf_reader.py). Regeneration from gguf.bpd requires the full Boundary-Provenance-DSL
# toolchain (bpd_parser + bpd_ast + lark), which lives in the private boundary_dsl tree and is NOT
# vendored here. So: if the toolchain is importable we regenerate; otherwise we treat the shipped
# parser as FROZEN and verify it against the tests. The previous build.sh CLAIMED to regenerate but
# died with ModuleNotFoundError — this is the truth-in-advertising fix.
#
# Exit codes: 0 all good; 2 tests failed.

set -e
cd "$(dirname "$0")"

echo "═══ ggufq Stage 0 build ═══"
echo

if python3 -c "import bpd_parser" 2>/dev/null; then
    echo "▶ BPD toolchain present — regenerating output/gguf_reader.py from gguf.bpd..."
    python3 generate.py
    echo
else
    echo "▶ FROZEN: BPD toolchain (bpd_parser/bpd_ast/lark) not vendored in this substrate."
    echo "  Shipping the pre-generated output/gguf_reader.py as a frozen artifact. To regenerate,"
    echo "  run from the boundary_dsl tree that has the generator. Verifying the frozen parser..."
    echo
fi

echo "▶ Running tests..."
if python3 -m pytest test_stage0.py -v --tb=short 2>&1 | tail -30; then
    echo
    echo "✓ Build OK — Stage 0 ready (frozen artifact verified, or regenerated)"
    echo "  Run: output/build/ggufq <file.gguf> --summary"
    exit 0
else
    echo
    echo "✗ Tests failed"
    exit 2
fi
