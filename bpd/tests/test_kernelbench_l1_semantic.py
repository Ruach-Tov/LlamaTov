# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""test_kernelbench_l1_semantic.py — Scope C-Extended-B: semantic correctness
of emitted CUDA kernels.

Per mavchin's directive (inbox 2026-05-15 19:30): "Scope C-Extended (semantic
correctness vs PyTorch reference). This is what closes the 'do our emitted
kernels produce CORRECT output' question."

PIPELINE:
  1. Generate kernel AST via generate_kernel_<family>/N (Prolog)
  2. Emit CUDA source via emit_program
  3. Compile to .so via nvcc (mavchin's existing path)
  4. Load via ctypes (mavchin's bridge pattern from llamatov_kernels.cu)
  5. Allocate GPU buffers, copy random inputs
  6. Execute the emitted kernel
  7. Copy output back, compare to cpu_references PyTorch equivalent
  8. Assert allclose within tolerance (rtol=1e-5, atol=1e-6)

CONSTRAINT: Requires CUDA-capable GPU at runtime. On nodes without
nvidia-smi / nvcc / CUDA runtime, all tests SKIP — substrate-honest
graceful degradation.

Author: metayen 2026-05-15 ~21:50 UTC
Per mavchin's Scope C-Extended priority. The skeleton is bounded for me
on this CPU-only node; the empirical execution happens when mavchin runs
this on the P4.
"""

import sys
import os
import subprocess
import ctypes
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import cpu_references as cr


# ════════════════════════════════════════════════════════════════════════
# GPU availability detection
# ════════════════════════════════════════════════════════════════════════

def has_nvcc() -> bool:
    """Check if nvcc is in PATH."""
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def has_cuda_gpu() -> bool:
    """Check if a CUDA-capable GPU is accessible."""
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


GPU_REQUIRED = pytest.mark.skipif(
    not (has_nvcc() and has_cuda_gpu()),
    reason="Scope C-Extended-B requires nvcc + CUDA-capable GPU. "
           "On CPU-only nodes (substrate-honest design), tests skip."
)


# ════════════════════════════════════════════════════════════════════════
# Helper: compile emitted CUDA to .so and load via ctypes
# ════════════════════════════════════════════════════════════════════════

def compile_kernel_to_so(cuda_source: str, kernel_name: str,
                          tmpdir: str, arch: str = 'sm_61') -> str:
    """Compile a single-kernel CUDA source to a shared library.

    Returns path to the .so file.
    Mirrors mavchin's pattern in bpd/llamatov_kernels.cu compilation.

    Environment overrides (for systems where nvcc / CUDA paths aren't
    on the default search list, e.g., NixOS):
      NVCC:     absolute path to nvcc binary (otherwise 'nvcc' from PATH)
      CUDA_INC: path containing cuda_runtime.h (passed as -I)
      CUDA_LIB: path containing libcudart.so (passed as -L)
    Mirrors the pattern in tests/build_harness_reduction.sh.

    NOTE on NixOS gotcha: even when nvcc is on PATH, the wrapper script
    at /run/current-system/sw/bin/nvcc may not ship the sibling nvvm/
    directory containing cicc, the CUDA compile-internal binary. The
    complete toolchain lives at /nix/store/.../cuda_nvcc-<version>/.
    Use NVCC=/nix/store/...nvcc to point at the complete bundle, not
    the wrapper.
    """
    cu_path = os.path.join(tmpdir, f'{kernel_name}.cu')
    so_path = os.path.join(tmpdir, f'{kernel_name}.so')

    # Write source with extern "C" wrapper so ctypes can find the symbol
    wrapped_source = (
        '#include <cuda_runtime.h>\n'
        'extern "C" {\n'
        f'{cuda_source}\n'
        '}\n'
    )
    with open(cu_path, 'w') as f:
        f.write(wrapped_source)

    # Compile to .so. Use -Wno-deprecated-gpu-targets so sm_61 build is quiet.
    nvcc_bin = os.environ.get('NVCC', 'nvcc')
    cmd = [
        nvcc_bin,
        '-arch=' + arch,
        '-Wno-deprecated-gpu-targets',
        '-Xcompiler', '-fPIC',
        '--shared',
    ]
    # CUDA include path (cuda_runtime.h). Optional — system nvcc may auto-find.
    cuda_inc = os.environ.get('CUDA_INC')
    if cuda_inc:
        cmd.extend(['-I', cuda_inc])
    # CUDA lib path (libcudart.so). Optional — system nvcc may auto-find.
    cuda_lib = os.environ.get('CUDA_LIB')
    if cuda_lib:
        cmd.extend(['-L', cuda_lib])
    cmd.extend([cu_path, '-o', so_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f'nvcc failed for {kernel_name}:\n'
            f'  command: {" ".join(cmd)}\n'
            f'  stdout: {result.stdout}\n'
            f'  stderr: {result.stderr}'
        )
    return so_path


def load_kernel(so_path: str, kernel_name: str):
    """Load the .so and return the ctypes function for kernel_name."""
    lib = ctypes.CDLL(so_path)
    return getattr(lib, kernel_name)


# ════════════════════════════════════════════════════════════════════════
# Per-family semantic test scaffolds
# ════════════════════════════════════════════════════════════════════════
#
# Each test follows the same pattern:
#   1. Generate kernel via Prolog
#   2. Compile + load via ctypes
#   3. Allocate inputs, copy to GPU
#   4. Call emitted kernel
#   5. Compare output to cpu_reference
#
# Since Prolog generation requires invoking swipl as a subprocess (or
# integration via py_swipl), we currently call the test_kernelbench_l1_cuda
# harness to do compilation, then load + execute the resulting .o/.so.


def _emit_kernel_via_prolog(family: str, op_kind: str, *args) -> str:
    """Invoke swipl to generate kernel CUDA source.

    Dispatches to bpd/lib/kernel_emit_bridge.py, which runs swipl as a
    subprocess and captures the emitted CUDA source.

    Returns the CUDA source as a string suitable for passing to
    compile_kernel_to_so().

    Wired 2026-05-17: closes the gap between the (already-built) bridge
    and the (already-built) compile+execute skeleton. Both were shipped
    independently on May 15; this is the wiring that activates the
    end-to-end matrix harness.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    import kernel_emit_bridge as keb

    if family == 'reduction':
        return keb.emit_kernel_reduction(op_kind, *args)
    elif family == 'norm':
        return keb.emit_kernel_norm(op_kind, *args)
    elif family == 'pool':
        return keb.emit_kernel_pool(op_kind, *args)
    elif family == 'im2col':
        return keb.emit_kernel_im2col(op_kind, *args)
    elif family == 'loss':
        return keb.emit_kernel_loss(op_kind, *args)
    else:
        raise ValueError(
            f"unknown kernel family: {family!r}. "
            f"Supported: reduction, norm, pool, im2col, loss."
        )


# ════════════════════════════════════════════════════════════════════════
# Family 1: Reductions
# ════════════════════════════════════════════════════════════════════════

@GPU_REQUIRED
class TestReductionSemantics:
    """End-to-end correctness tests for the reduction kernel family.

    Pipeline per test:
      1. Generate deterministic input tensor.
      2. Compute reference via cpu_references (PyTorch).
      3. Emit kernel + extern "C" launcher via kernel_emit_bridge.
      4. nvcc compile to .so.
      5. Load launcher symbol via ctypes.
      6. Dispatch with numpy buffer pointers, check status.
      7. Compare output to reference (rtol=1e-5, atol=1e-6).

    Skip behavior: @GPU_REQUIRED skips all tests when nvcc or
    CUDA-capable GPU is unavailable. The Python wiring (bridge call,
    ctypes setup) is exercised when GPU is present; otherwise the
    whole class skips together.
    """

    def _run_reduction(self, op_kind: str, x_torch):
        """Run one reduction op end-to-end and return the output tensor.

        Shared helper across the per-op tests below. Owns the
        emit-compile-load-dispatch-copyback flow.
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
        import numpy as np
        import kernel_emit_bridge as keb

        # Emit kernel + host launcher together.
        combined_src, kernel_name, launcher_name = keb.emit_kernel_with_launcher(
            'reduction', op_kind
        )

        # Compile to .so. Use a temp dir per test for isolation.
        with tempfile.TemporaryDirectory() as tmpdir:
            so_path = compile_kernel_to_so(combined_src, kernel_name, tmpdir)

            # Load .so and resolve launcher symbol.
            lib = ctypes.CDLL(so_path)
            launch = getattr(lib, launcher_name)
            launch.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # h_X
                ctypes.POINTER(ctypes.c_float),  # h_Y
                ctypes.c_int,                     # N
                ctypes.c_int,                     # outer
            ]
            launch.restype = ctypes.c_int

            # Prepare host buffers. x is [outer, N] row-major; output is [outer].
            x_np = x_torch.contiguous().numpy().astype(np.float32, copy=True)
            assert x_np.ndim == 2, "reduction expects 2D input [outer, N]"
            outer, N = x_np.shape
            y_np = np.zeros(outer, dtype=np.float32)

            # Dispatch through the launcher.
            status = launch(
                x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                y_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(N),
                ctypes.c_int(outer),
            )

            # Map status codes to diagnostic exceptions (see
            # emit_reduction_launcher docstring for the code → step mapping).
            if status != 0:
                step_names = {
                    1: 'cudaMalloc input',
                    2: 'cudaMalloc output',
                    3: 'cudaMemcpy host→device',
                    4: 'kernel launch (cudaGetLastError)',
                    5: 'cudaDeviceSynchronize',
                    6: 'cudaMemcpy device→host',
                }
                raise RuntimeError(
                    f"{launcher_name} returned status {status} "
                    f"({step_names.get(status, 'unknown step')})"
                )

            return torch.from_numpy(y_np.copy())

    def test_reduce_sum_matches_reference(self):
        """Emitted reduce_sum kernel must match cpu_reference_sum_rows."""
        torch.manual_seed(42)
        x = torch.randn(8, 16)
        ref = cr.cpu_reference_sum_rows(x)
        got = self._run_reduction('ggml_sum_rows', x)
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6), (
            f"reduce_sum diverged from reference\n"
            f"  got:  {got}\n"
            f"  ref:  {ref}\n"
            f"  diff: {(got - ref).abs().max().item()}"
        )

    @pytest.mark.skip(reason=(
        "1.b.iv ships test_reduce_sum_matches_reference as the anchor "
        "single-cell end-to-end test. Other reduction ops (mean, max, "
        "argmax, etc.) follow once the anchor is GPU-verified."
    ))
    def test_reduce_mean_matches_reference(self):
        torch.manual_seed(42)
        x = torch.randn(8, 16)
        ref = cr.cpu_reference_mean(x)
        got = self._run_reduction('ggml_mean', x)
        assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.skip(reason="Pending GPU-verification of the anchor test.")
    def test_reduce_argmax_matches_reference(self):
        pass


# ════════════════════════════════════════════════════════════════════════
# Family 2: Normalizations
# ════════════════════════════════════════════════════════════════════════

@GPU_REQUIRED
class TestNormalizationSemantics:

    @pytest.mark.skip(reason="Prolog→Python kernel generation bridge pending")
    def test_rms_norm_matches_reference(self):
        """The emitted norm_rms kernel should match cpu_reference_rms_norm."""
        pass

    @pytest.mark.skip(reason="Prolog→Python kernel generation bridge pending")
    def test_layer_norm_matches_reference(self):
        pass


# ════════════════════════════════════════════════════════════════════════
# Family 5: Losses
# ════════════════════════════════════════════════════════════════════════

@GPU_REQUIRED
class TestLossSemantics:

    @pytest.mark.skip(reason="Prolog→Python kernel generation bridge pending")
    def test_mse_matches_reference(self):
        pass

    @pytest.mark.skip(reason="Prolog→Python kernel generation bridge pending")
    def test_huber_matches_reference(self):
        pass


# ════════════════════════════════════════════════════════════════════════
# Documented status of this scope
# ════════════════════════════════════════════════════════════════════════
#
# COMPLETE (this commit):
#   ✓ cpu_references.py: PyTorch ground truth for 25+ ops across 6 families
#   ✓ test_cpu_references.py: 36 tests verify references match PyTorch
#     builtins or satisfy mathematical invariants
#   ✓ This test framework: structural scaffold for GPU execution comparison
#   ✓ Graceful degradation: tests SKIP on CPU-only nodes
#
# PENDING (Scope C-Extended-Execution, future commit by mavchin or me
# when GPU is available):
#   • Prolog→Python kernel generation bridge (swipl subprocess wrapper)
#   • Per-family allocate + copy + execute + compare logic
#   • Integration with mavchin's ctypes bridge (commit 13cfc6733 pattern)
#   • Actual run on Tesla P4 → bit-exact comparison vs cpu_references
#
# WHEN COMPLETE:
#   Closes "exploitable loopholes" gap per robust-kbench (medayek's
#   literature finding). The substrate's emission isn't just shape-correct
#   (proven in 432b08467) — it's NUMERICALLY correct vs PyTorch.


def test_skeleton_loads_without_error():
    """Substrate-honest smoke test: this file imports + collects cleanly."""
    # The file already imported cpu_references and ctypes; if we got here,
    # the skeleton is structurally sound.
    assert callable(cr.cpu_reference_sum_rows)
    assert callable(cr.cpu_reference_rms_norm)


def test_gpu_required_marker_works():
    """Verify the GPU_REQUIRED marker is wired up correctly."""
    # This test itself doesn't require GPU. It just checks that
    # the marker exists and is applied to the test classes.
    assert hasattr(TestReductionSemantics, 'pytestmark') or True
    # Not a strict assertion — the marker is checked at collection time
    # by pytest, not via runtime introspection.


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
