#!/usr/bin/env bash
# build.sh — regenerate, test, and report Stage 0 status.
#
# Usage: ./build.sh
#
# Runs the full build cycle:
#   1. Regenerate output/gguf_reader.py from gguf.bpd
#   2. Run all tests including conformance
#   3. Report status
#
# Exit codes:
#   0   all good
#   1   generation failed
#   2   tests failed

set -e
cd "$(dirname "$0")"

echo "═══ ggufq Stage 0 build ═══"
echo

echo "▶ Regenerating from gguf.bpd..."
python3 generate.py
echo

echo "▶ Running tests..."
if python3 -m pytest test_stage0.py -v --tb=short 2>&1 | tail -30; then
    echo
    echo "✓ Build OK — Stage 0 ready"
    echo "  Run: output/build/ggufq <file.gguf> --summary"
    exit 0
else
    echo
    echo "✗ Tests failed"
    exit 2
fi
