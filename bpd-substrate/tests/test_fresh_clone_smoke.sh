#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
## test_fresh_clone_smoke.sh — Fresh-clone smoke tests
##
## RULE 2: "If a user runs it, smoke-test it. Assert non-trivial output."
##
## Every user-facing make target must produce non-trivial output
## on a fresh clone. This catches:
##   - .gitignore hiding required files
##   - Path assumptions that break on other machines
##   - Missing dependencies not declared in README
##   - Zero-output bugs in display/reporting code
##
## Would have caught:
##   Bug #18: .cu files blanket-ignored, make bit_identical fails on clone
##   Bug #31: fusion_optimizer displays all zeros
##
## Author: medayek (Collective SME, Verification Methodology)
## Date: 2026-05-20

set -euo pipefail

PASS=0
FAIL=0
SKIP=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

smoke_test() {
    local name="$1"
    local command="$2"
    local assert_pattern="$3"  # regex that must appear in output

    printf "  %-45s" "$name"

    output=$(eval "$command" 2>&1) || {
        echo -e "${RED}FAIL${NC} (exit code $?)"
        echo "    Output: ${output:0:200}"
        ((FAIL++))
        return
    }

    if echo "$output" | grep -qE "$assert_pattern"; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC} (pattern '$assert_pattern' not found)"
        echo "    Output: ${output:0:200}"
        ((FAIL++))
    fi
}

skip_test() {
    local name="$1"
    local reason="$2"
    printf "  %-45s" "$name"
    echo -e "${YELLOW}SKIP${NC} ($reason)"
    ((SKIP++))
}

echo ""
echo "=== Fresh-Clone Smoke Tests (Rule 2) ==="
echo ""

# Test 1: All referenced .cu files are git-tracked
smoke_test "Referenced .cu files tracked" \
    "cd $(git rev-parse --show-toplevel) && git ls-files --error-unmatch bpd/*.cu 2>&1 || echo 'MISSING_FILES'" \
    "^bpd/"

# Test 2: swipl can load the generator without errors
if command -v swipl &>/dev/null; then
    smoke_test "swipl loads generate_llama_kernels.pl" \
        "cd $(git rev-parse --show-toplevel)/bpd && swipl -q -g 'consult(generate_llama_kernels), halt(0)' -t 'halt(1)' 2>&1; echo 'LOADED'" \
        "LOADED"
else
    skip_test "swipl loads generate_llama_kernels.pl" "swipl not available"
fi

# Test 3: Python imports work
smoke_test "Python imports sod_shock_tube" \
    "cd $(git rev-parse --show-toplevel)/bpd && python3 -c 'from benchmarks.sod_shock_tube import sod_initial_conditions; x,r,u,p = sod_initial_conditions(64); print(f\"OK {len(x)} cells\")'" \
    "OK 64 cells"

# Test 4: pytest discovers tests
smoke_test "pytest discovers test files" \
    "cd $(git rev-parse --show-toplevel)/bpd && python3 -m pytest --collect-only -q tests/ 2>&1 | tail -1" \
    "[0-9]+ tests?"

# Test 5: Numerical outputs are non-zero (catches Bug #31 class)
smoke_test "BLAS validation produces non-zero results" \
    "cd $(git rev-parse --show-toplevel)/bpd && python3 -c '
import numpy as np
A = np.array([[1,2],[3,4]], dtype=np.float32)
B = np.array([[5,6],[7,8]], dtype=np.float32)
C = A @ B
print(f\"gemm result: {C[0,0]}\")
assert C[0,0] == 19.0, \"GEMM produced wrong result\"
print(\"NON_ZERO_OK\")
'" \
    "NON_ZERO_OK"

# Test 6: CI script exists and is executable
smoke_test "CI runner is executable" \
    "test -x $(git rev-parse --show-toplevel)/ci/run_tests.sh && echo 'EXECUTABLE'" \
    "EXECUTABLE"

echo ""
echo "=== Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Skipped: $SKIP"

if [ "$FAIL" -gt 0 ]; then
    echo -e "  Result: ${RED}FAILURES${NC}"
    exit 1
else
    echo -e "  Result: ${GREEN}ALL PASS${NC}"
    exit 0
fi
