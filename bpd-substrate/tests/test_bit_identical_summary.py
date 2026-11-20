"""Unit tests for bit_identical.py's summary rendering.

Tests the _render_summary(results) helper with mock results lists.
No CUDA required — pure output-format verification.

Motivated by ColonistOne PR#7 catch (2026-09-05): the previous single
"PASSED: N/M" line was OR across two contracts and over-claimed for
the bit-identical question. Now each contract gets its own loud
verdict. These tests verify the new output distinguishes contracts
correctly.
"""
import os
import sys

# Make bench/ importable so we can import bit_identical's helper.
# The test file lives at bpd-substrate/tests/, the module at
# bpd-substrate/bench/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "bench"))
sys.path.insert(0, BENCH_DIR)


def _import_helper():
    """Import _render_summary. Done inside a function so pytest collection
    doesn't require torch+CUDA (which bit_identical.py imports at top).

    We stub torch away for import time so this test runs on machines
    without CUDA — the summary helper is pure Python (string formatting
    over the results list) and never touches torch itself. Stubbing is
    safe because the failure mode we'd catch (someone adding a torch
    dependency inside _render_summary) would surface immediately in a
    real bit_identical.py run rather than sneaking through this test.

    Per Medayek ef43ac31 review: keep the stubbing, document the reason.
    Alternative (moving torch import inside main()) would be preferable
    structurally but is a larger refactor beyond ColonistOne's scope.
    """
    import types
    # Stub torch + torch.nn.functional so import succeeds without CUDA
    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: True)
    torch_stub.__version__ = "stub"
    F_stub = types.ModuleType("torch.nn.functional")
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
    sys.modules.setdefault("torch.nn.functional", F_stub)
    import bit_identical
    return bit_identical._render_summary


# Results tuple format: (kernel, shape, status_cublas, tag_cublas,
#                        status_truth, tag_truth, passed)

def _make_result(status_cublas, status_truth, kernel="sgemm", shape="64x64"):
    """Helper to build a results tuple. `passed` is derived per the
    same OR logic as the real bit_identical code."""
    passed = (
        status_cublas in ('BIT_IDENTICAL', 'PASS_ABS_TOLERANCE')
        or status_truth in ('BIT_IDENTICAL_VS_TRUTH', 'WITHIN_ERROR_BOUND')
    )
    return (kernel, shape, status_cublas, "tag_cublas",
            status_truth, "tag_truth", passed)


def test_summary_replaces_ambiguous_passed_line():
    """The single ambiguous 'PASSED: N/M' line must not appear.
    It's been replaced by per-contract loud verdicts."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
        _make_result('BIT_IDENTICAL', 'WITHIN_ERROR_BOUND'),
    ]
    output = "\n".join(render(results))
    # The bare "PASSED: N/M" phrase must not be the loud verdict.
    # (It appears in "OVERALL PASSED (either contract)" but that's labeled.)
    assert "PASSED:" not in output, (
        "Ambiguous 'PASSED: N/M' line should be replaced with per-"
        "contract verdicts. Found in output:\n" + output
    )


def test_summary_has_bit_identical_verdict_first():
    """The bit-identical verdict should be the FIRST loud line
    (answering ColonistOne's specific question directly)."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
        _make_result('BIT_IDENTICAL', 'WITHIN_ERROR_BOUND'),
    ]
    lines = render(results)
    # Find the first line that starts with a verdict prefix
    verdict_lines = [
        l for l in lines
        if l.startswith("BIT-IDENTICAL") or l.startswith("TRUTH-CONTRACT")
        or l.startswith("OVERALL")
    ]
    assert verdict_lines, "No verdict lines found in output"
    assert verdict_lines[0].startswith("BIT-IDENTICAL vs cuBLAS:"), (
        f"First verdict should be BIT-IDENTICAL, got: {verdict_lines[0]}"
    )


def test_summary_counts_bit_identical_correctly():
    """BIT-IDENTICAL count = n_bit_id + n_abs_tol (cuBLAS-matches).
    Truth-contract-only passes must NOT inflate this count."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),   # counts for bit-id
        _make_result('BIT_IDENTICAL', 'WITHIN_ERROR_BOUND'),       # counts for bit-id
        _make_result('PASS_ABS_TOLERANCE', 'WITHIN_ERROR_BOUND'),  # counts for bit-id
        _make_result('WITHIN_ERROR_BOUND', 'WITHIN_ERROR_BOUND'),  # does NOT count for bit-id (truth-only pass)
        _make_result('WITHIN_ERROR_BOUND', 'WITHIN_ERROR_BOUND'),  # does NOT count for bit-id
    ]
    output = "\n".join(render(results))
    # 3 of 5 match cuBLAS bits
    assert "BIT-IDENTICAL vs cuBLAS: 3/5" in output, (
        f"Expected 'BIT-IDENTICAL vs cuBLAS: 3/5' in output. Got:\n{output}"
    )
    # But 5 of 5 pass either contract
    assert "OVERALL PASSED (either contract): 5/5" in output


def test_summary_labels_overall_passed_explicitly():
    """OVERALL PASSED line must explicitly say '(either contract)' so
    the OR-semantics is not ambiguous."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
    ]
    output = "\n".join(render(results))
    assert "OVERALL PASSED (either contract):" in output, (
        f"OVERALL PASSED must be labeled as either-contract semantics. Got:\n{output}"
    )


def test_summary_keeps_teaching_parentheticals():
    """Per Iyun b8585588 coord review: teaching parentheticals explain
    what each contract means. External contributors shouldn't have to
    read source to understand the verdicts."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
    ]
    output = "\n".join(render(results))
    # BIT-IDENTICAL verdict should be self-explaining
    assert "0 ULP match to cuBLAS output bits" in output
    # TRUTH-CONTRACT verdict should name the error bound
    assert "within O(sqrt(K))*eps of f64 truth" in output


def test_summary_handles_all_pass_case():
    """When all results pass, output says 'ALL KERNELS PASS (either contract)'
    — labeled, not the ambiguous bare 'ALL KERNELS PASS'."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
        _make_result('BIT_IDENTICAL', 'WITHIN_ERROR_BOUND'),
    ]
    output = "\n".join(render(results))
    assert "ALL KERNELS PASS (either contract):" in output, (
        f"Expected labeled 'ALL KERNELS PASS (either contract):' in output. Got:\n{output}"
    )


def test_summary_handles_failure_case():
    """When some results fail both contracts, output includes FAIL: N
    and per-failure diagnostics."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
        _make_result('EXCEEDS_ERROR_BOUND', 'EXCEEDS_ERROR_BOUND', kernel='broken_kernel'),
    ]
    output = "\n".join(render(results))
    assert "FAIL: 1" in output
    assert "broken_kernel" in output
    assert "NEXT WORK ITEMS" in output


def test_summary_empty_gemm_cases_no_truth_contract_line():
    """If no GEMM cases (n_gemm_cases=0, all results have truth='N/A'),
    the TRUTH-CONTRACT line is omitted (not misleading with 0/0)."""
    render = _import_helper()
    results = [
        _make_result('BIT_IDENTICAL', 'N/A', kernel='non_gemm'),
    ]
    output = "\n".join(render(results))
    assert "TRUTH-CONTRACT passes:" not in output, (
        "Truth-contract line should not appear when no GEMM cases exist. "
        f"Got:\n{output}"
    )
    # But BIT-IDENTICAL should still show
    assert "BIT-IDENTICAL vs cuBLAS: 1/1" in output


def test_no_bare_passed_regression_canary():
    """REGRESSION CANARY (Medayek ef43ac31 review): no line in the
    summary output should start with a bare 'PASSED:' phrase or emit
    'PASSED' as a loud verdict header without contract qualification.

    This is stricter than test_summary_replaces_ambiguous_passed_line
    (which checks the specific string). This test verifies the
    INVARIANT: any 'PASSED' loud-verdict-style line must be qualified
    with a contract label. Catches future edits that reintroduce
    ambiguous verdict phrasing in a place the specific test didn't
    imagine.
    """
    import pytest
    render = _import_helper()
    # Multiple scenarios to force the code through various branches
    scenarios = [
        # All-pass, GEMM cases
        [_make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH') for _ in range(3)],
        # Mixed pass/fail
        [
            _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH'),
            _make_result('EXCEEDS_ERROR_BOUND', 'EXCEEDS_ERROR_BOUND'),
        ],
        # No GEMM cases (only non-GEMM kernels)
        [_make_result('BIT_IDENTICAL', 'N/A')],
    ]
    for i, results in enumerate(scenarios):
        output_lines = render(results)
        for line in output_lines:
            stripped = line.strip()
            # Any line starting with "PASSED" as a header (not embedded
            # in something like "OVERALL PASSED") must have a contract
            # qualifier. Loud verdicts should never start with bare
            # "PASSED:" — that's the anti-pattern ColonistOne caught.
            if stripped.startswith("PASSED:") or stripped.startswith("PASSED "):
                pytest.fail(
                    f"Scenario {i}: found bare 'PASSED' loud verdict "
                    f"line: {line!r}\nAny PASSED verdict must be "
                    f"qualified per-contract or labeled '(either "
                    f"contract)'. Full output:\n" +
                    "\n".join(output_lines)
                )


def test_summary_reproduces_colonistone_case():
    """Reproduce the ColonistOne case: 20 results, 9 bit-identical vs
    cuBLAS, 20 passing either contract. Verify the loud verdicts
    distinguish these correctly."""
    render = _import_helper()
    # 9 results that are BIT_IDENTICAL vs cuBLAS AND within truth bound
    results = [
        _make_result('BIT_IDENTICAL', 'BIT_IDENTICAL_VS_TRUTH', shape=f'{i}x{i}')
        for i in range(9)
    ]
    # 11 results that are WITHIN_ERROR_BOUND for truth but NOT bit-identical
    results += [
        _make_result('EXCEEDS_ERROR_BOUND', 'WITHIN_ERROR_BOUND', shape=f'{i}x{i}')
        for i in range(9, 20)
    ]
    output = "\n".join(render(results))
    # 9 of 20 match cuBLAS bits — the honest answer to "am I bit-identical?"
    assert "BIT-IDENTICAL vs cuBLAS: 9/20" in output
    # 20 of 20 pass truth contract
    assert "TRUTH-CONTRACT passes:   20/20" in output
    # 20 of 20 pass either contract (the OR-aggregate)
    assert "OVERALL PASSED (either contract): 20/20" in output
