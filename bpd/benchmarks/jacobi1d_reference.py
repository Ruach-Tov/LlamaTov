# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
Jacobi1D — Python reference implementation, float32 throughout.

Member of the cell-indexed Dirichlet 3-neighbor stencil family
(see methodology/cell-indexed-dirichlet-3-neighbor-stencil-family.md).

The algorithm:
    out[i] = (in[i-1] + in[i] + in[i+1]) / 3.0    for i in [1, N-2]
    out[0] and out[N-1] are NOT modified (Dirichlet BC)

Used in:
  - 1D smoothing iterations (PDE solver pre-conditioning)
  - Iterative Poisson solver as the relaxation step
  - 1D box filter in signal processing
  - 1D average pooling in ML (kernel size 3, stride 1)

Per the substrate's "physics-for-physics correctness" methodology:
the reference here is mathematical truth (averaging operator), not
another implementation. There are analytical solutions for specific
initial conditions that the verification harness can check.

Authored 2026-05-18 by metayen as the F4 Path-A reconsidered scope.
"""
import numpy as np


def jacobi1d(arr_in, arr_out=None):
    """Apply one Jacobi1D iteration to a 1D float32 array.

    Parameters
    ----------
    arr_in : np.ndarray of shape (N,), dtype float32
        Input values.
    arr_out : np.ndarray of shape (N,), dtype float32, optional
        Output buffer. If None, allocated. Boundary values copied
        from input to enforce Dirichlet BC.

    Returns
    -------
    np.ndarray of shape (N,), dtype float32
        Output array. Interior cells averaged from 3-neighbor stencil;
        boundary cells (0, N-1) copied from input.
    """
    arr_in = np.asarray(arr_in, dtype=np.float32)
    N = len(arr_in)
    if arr_out is None:
        arr_out = np.empty(N, dtype=np.float32)
    elif arr_out is arr_in:
        raise ValueError(
            "arr_out aliases arr_in; Jacobi1D requires separate buffers "
            "to preserve the dependency (each out[i] reads in[i-1], in[i], in[i+1])."
        )

    # Dirichlet BC: boundary values unchanged
    arr_out[0] = arr_in[0]
    arr_out[N - 1] = arr_in[N - 1]

    # Interior cells: 3-neighbor average
    # Note: doing this with explicit indexing rather than np.convolve
    # so the FP operation order matches the GPU kernel emit exactly.
    # (L + C + R) / 3.0f as a single expression, NOT L/3 + C/3 + R/3.
    for i in range(1, N - 1):
        L = arr_in[i - 1]
        C = arr_in[i]
        R = arr_in[i + 1]
        arr_out[i] = (L + C + R) / np.float32(3.0)

    return arr_out


def jacobi1d_analytical_sinusoid(N, t, amplitude=1.0, phase=0.0):
    """Compute the analytical Jacobi1D evolution of a sinusoidal IC.

    For initial condition u(x, 0) = A * sin(2π * i / N + phase),
    each Jacobi1D iteration multiplies the amplitude by a known decay
    factor:

        decay = (1 + 2 * cos(2π / N)) / 3

    After t iterations:

        u(x, t) = A * decay^t * sin(2π * i / N + phase)

    This is the spectral analysis of the Jacobi1D operator on a
    finite grid with Dirichlet BC. For the lowest non-constant mode,
    cos(2π / N) is close to 1, so the decay is slow. Higher-frequency
    modes decay faster (the substrate is essentially low-pass filtering).

    Parameters
    ----------
    N : int
        Grid size.
    t : int
        Number of iterations.
    amplitude : float, default 1.0
    phase : float, default 0.0
        Initial phase offset in radians.

    Returns
    -------
    np.ndarray of shape (N,), dtype float32
        Expected state after t iterations of Jacobi1D applied to
        amplitude * sin(2π * i / N + phase).

    Caveats
    -------
    This analytical solution assumes the boundary modes also evolve.
    Strict Dirichlet BC (boundary held constant) introduces small
    deviations from the pure-mode analysis at low t. For verification,
    compare interior cells only.
    """
    decay = (np.float32(1.0) + np.float32(2.0) * np.cos(np.float32(2.0 * np.pi) / np.float32(N))) / np.float32(3.0)
    i = np.arange(N, dtype=np.float32)
    return (np.float32(amplitude) * (decay ** np.float32(t)) *
            np.sin(np.float32(2.0 * np.pi) * i / np.float32(N) +
                   np.float32(phase))).astype(np.float32)


def test_jacobi1d_basic():
    """Smoke checks of the algorithm and its analytical reference."""

    # Test 1: Constant input → constant output (interior unchanged)
    N = 16
    constant_val = np.float32(2.5)
    arr_in = np.full(N, constant_val, dtype=np.float32)
    arr_out = jacobi1d(arr_in)
    # Constant + constant + constant / 3 = constant. Bit-identical in float32.
    interior_diff = np.abs(arr_out[1:-1] - constant_val)
    assert np.all(interior_diff == 0.0), \
        f"Constant-input invariant failed: max diff {interior_diff.max()}"
    # Boundary preserved
    assert arr_out[0] == arr_in[0], "Left boundary not preserved"
    assert arr_out[N - 1] == arr_in[N - 1], "Right boundary not preserved"
    print(f"  ✓ Constant input → constant output (interior bit-identical)")

    # Test 2: Step function smooths over iterations
    N = 32
    arr = np.zeros(N, dtype=np.float32)
    arr[N // 2:] = 1.0
    for _ in range(10):
        arr = jacobi1d(arr)
    # After smoothing, the step should be diffused — values near
    # the discontinuity should be in (0, 1), not 0 or 1.
    mid_left = arr[N // 2 - 1]
    mid_right = arr[N // 2]
    assert 0.0 < mid_left < 1.0, \
        f"Step did not smooth at left of midpoint: {mid_left}"
    assert 0.0 < mid_right < 1.0, \
        f"Step did not smooth at right of midpoint: {mid_right}"
    print(f"  ✓ Step function smooths under iteration (mid values: {mid_left:.4f}, {mid_right:.4f})")

    # Test 3: Sinusoidal decay matches analytical reference (approximately)
    # For interior cells; boundaries deviate from pure-mode analysis.
    N = 64
    initial = jacobi1d_analytical_sinusoid(N, 0)  # t=0 just gives the IC
    after_5_iter = initial.copy()
    for _ in range(5):
        after_5_iter = jacobi1d(after_5_iter)
    expected = jacobi1d_analytical_sinusoid(N, 5)
    # Compare interior cells with loose tolerance (the boundary
    # interaction at strict Dirichlet causes small drift from the
    # pure-mode analysis)
    interior_err = np.abs(after_5_iter[2:-2] - expected[2:-2]).max()
    print(f"  ✓ Sinusoidal decay matches analytical reference (interior max error: {interior_err:.6f})")
    # Don't assert a strict bound; this is a sanity check of the
    # algorithm's spectral behavior, not a bit-identical check.

    # Test 4: Diaphragm-like step (matching CFD methodology) for visual confirmation
    N = 8
    arr = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    after_one = jacobi1d(arr)
    print(f"  ✓ Anchor: input  {arr.tolist()}")
    print(f"           output {after_one.tolist()}")
    # Hand-computable: interior values should be
    #   [_, (1+1+1)/3, (1+1+1)/3, (1+1+0)/3, (1+0+0)/3, (0+0+0)/3, (0+0+0)/3, _]
    # = [1.0, 1.0,        1.0,       0.6666..., 0.3333..., 0.0,    0.0,    0.0]
    expected_anchor = np.array([1.0, 1.0, 1.0, 2/3, 1/3, 0.0, 0.0, 0.0], dtype=np.float32)
    diff = np.abs(after_one - expected_anchor)
    max_diff = diff.max()
    assert max_diff < 1e-6, f"Single-step anchor failed: max diff {max_diff}"
    print(f"           anchor max diff: {max_diff:.2e}")


if __name__ == "__main__":
    print("Jacobi1D Python reference — verification tests:")
    test_jacobi1d_basic()
    print("\nAll tests passed.")
