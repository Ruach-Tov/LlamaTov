# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""kernel_emit_bridge.py — Python↔Prolog kernel emission bridge.

Per mavchin's directive (inbox 19:42:46): option (b) — write the swipl
subprocess bridge as a separate commit; they run it on P4.

This module wraps the Prolog kernel generation predicates as Python
callables. Each `emit_kernel_<family>(...)` invokes swipl as a
subprocess, runs the appropriate generate_kernel_<family>/N goal,
captures the emitted CUDA source, and returns it as a string.

USAGE:
  from kernel_emit_bridge import emit_kernel_reduction, emit_kernel_norm
  cuda_src = emit_kernel_reduction('ggml_sum_rows')
  # → "#include <cuda_runtime.h>\\n\\n__global__ void reduce_sum(...)..."

The resulting source can be:
  1. Compiled via nvcc to .so (per Scope C: 432b08467)
  2. Loaded via ctypes (per mavchin's bridge pattern: 13cfc6733)
  3. Executed on GPU with allocated buffers
  4. Compared against cpu_references.py (per Scope C-Extended-A: 608f3b346)

That is Scope C-Extended-B: the empirical semantic correctness loop.

Author: metayen 2026-05-15
Per mavchin's option (b) directive. CPU-only implementation — testable
on this node; doesn't require nvcc or GPU. The compile+execute steps
are downstream consumers' responsibility.
"""

import os
import subprocess
import shutil
from typing import Optional


# ════════════════════════════════════════════════════════════════════════
# Constants and helpers
# ════════════════════════════════════════════════════════════════════════

# Path to the bpd directory (where lib/ and tests/ live)
_BPD_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))


def _swipl_available() -> bool:
    """Check if swipl is in PATH."""
    return shutil.which('swipl') is not None


class KernelEmitError(Exception):
    """Raised when swipl invocation or AST emission fails."""
    pass


def _run_swipl_emit(prolog_goal: str, timeout: float = 30.0) -> str:
    """Run swipl with the given goal, return captured CUDA source.

    The goal MUST include `emit_program([... K], C), write(C), halt`
    to produce the CUDA source on stdout.

    Stderr is suppressed to filter out discontiguous-predicate warnings
    that aren't actionable for the consumer.

    Raises KernelEmitError on subprocess failure or empty output.
    """
    if not _swipl_available():
        raise KernelEmitError(
            "swipl not in PATH. Install SWI-Prolog or run inside a "
            "nix-shell with swiProlog. On this node:\n"
            "  nix-shell -p swiProlog --run 'python3 your_script.py'"
        )

    # Construct command: swipl -q -g <goal>
    # Run from bpd/ so relative module paths (lib/c_ast, lib/kernel_templates)
    # resolve correctly.
    cmd = ['swipl', '-q', '-g', prolog_goal]

    try:
        result = subprocess.run(
            cmd,
            cwd=_BPD_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise KernelEmitError(
            f"swipl invocation timed out after {timeout}s. Goal: {prolog_goal}"
        ) from e

    if result.returncode != 0:
        raise KernelEmitError(
            f"swipl exited with code {result.returncode}.\n"
            f"  goal: {prolog_goal}\n"
            f"  stdout: {result.stdout!r}\n"
            f"  stderr: {result.stderr!r}"
        )

    # swipl emits warnings/dibbur-environment-loaded to stderr; CUDA source
    # to stdout via our `write(C), halt` pattern.
    cuda_source = result.stdout

    # Sanity: the source should start with #include or __global__ or similar.
    # If it's empty or contains only whitespace, that's a failure.
    if not cuda_source.strip():
        raise KernelEmitError(
            f"swipl emitted empty output for goal: {prolog_goal}\n"
            f"  stderr: {result.stderr!r}"
        )

    return cuda_source


def _construct_goal(generate_call: tuple,
                    module: str = 'lib/kernel_templates',
                    includes: tuple = ('cuda_runtime.h',)) -> str:
    """Wrap a generate_kernel_<family>(...) call with emit_program + halt.

    Builds the goal as a Prolog AST (via lib/prolog_goal.py), then
    serializes once at the boundary. The substrate-honest equivalent
    of c_ast for Prolog goal construction.

    Args:
        generate_call: a Prolog goal AST term (built via pg_call from
            lib/prolog_goal). Must bind K (the kernel AST) in its
            last argument so emit_program receives the kernel to render.
        module: the Prolog module to use_module that provides the
            generator. Default 'lib/kernel_templates' for the existing
            reduction/norm/pool/im2col/loss families. Use
            'lib/kernel_templates_llama' for the activation /
            binary-elementwise families (shipped in f6695d2e1).
        includes: tuple of system header names to emit as c_include_sys
            preamble. The default ('cuda_runtime.h',) is sufficient for
            reductions; activations need ('cuda_runtime.h', 'math.h')
            because their bodies use expf/erff/tanhf/fmaxf.

    Returns:
        Canonical Prolog goal string suitable for `swipl -g <result>`.
    """
    from prolog_goal import (
        pg_atom, pg_string, pg_var, pg_call, pg_list, pg_seq, serialize,
    )

    # Build the include list as Prolog terms.
    include_terms = [
        pg_call('c_include_sys', [pg_string(h)]) for h in includes
    ] + [pg_atom('c_blank'), pg_var('K')]

    goal = pg_seq([
        pg_call('use_module', [pg_atom('lib/c_ast')]),
        pg_call('use_module', [pg_atom(module)]),
        generate_call,
        pg_call('emit_program', [pg_list(include_terms), pg_var('C')]),
        pg_call('write', [pg_var('C')]),
        pg_atom('halt'),
    ])
    return serialize(goal)


# ════════════════════════════════════════════════════════════════════════
# Family 1: Reductions
# ════════════════════════════════════════════════════════════════════════
#
# Mirrors generate_kernel_reduction/4(+OpKind, +Dim, +ReductionMode, -K)

def emit_kernel_reduction(op_kind: str, dim: int = 2,
                           reduction_mode: str = 'axis_inner') -> str:
    """Emit CUDA for a reduction kernel.

    op_kind: ggml_sum_rows, ggml_mean, ggml_max, ggml_min,
             ggml_argmax, ggml_argmin, ggml_cumsum, ggml_cumprod
    dim: input dimensionality (reserved at codegen; runtime determines N)
    reduction_mode: axis_inner (default; only mode L1 exercises)
    """
    from prolog_goal import pg_atom, pg_int, pg_var, pg_call
    call = pg_call('generate_kernel_reduction', [
        pg_atom(op_kind),
        pg_int(dim),
        pg_atom(reduction_mode),
        pg_var('K'),
    ])
    return _run_swipl_emit(_construct_goal(call))


# ════════════════════════════════════════════════════════════════════════
# Family 2: Normalizations
# ════════════════════════════════════════════════════════════════════════
#
# Mirrors generate_kernel_norm/4(+OpKind, +Dim, +Affine, -K)

def emit_kernel_norm(op_kind: str, dim: int = 2,
                     affine: bool = False) -> str:
    """Emit CUDA for a normalization kernel.

    op_kind: ggml_norm, ggml_l2_norm, ggml_rms_norm, ggml_group_norm
    dim: input dimensionality (reserved at codegen)
    affine: whether to include W (and B for layer) learned parameters
    """
    from prolog_goal import pg_atom, pg_int, pg_var, pg_call
    call = pg_call('generate_kernel_norm', [
        pg_atom(op_kind),
        pg_int(dim),
        pg_atom('true' if affine else 'false'),
        pg_var('K'),
    ])
    return _run_swipl_emit(_construct_goal(call))


# ════════════════════════════════════════════════════════════════════════
# Family 3: Pooling
# ════════════════════════════════════════════════════════════════════════
#
# Mirrors generate_kernel_pool/5(+OpKind, +Dim, +PoolKind, +Params, -K)

def emit_kernel_pool(op_kind: str, dim: int,
                      pool_kind: str = 'max') -> str:
    """Emit CUDA for a pooling kernel.

    op_kind: ggml_pool_1d, ggml_pool_2d, ggml_pool_3d
    dim: 1, 2, or 3
    pool_kind: max or avg
    """
    from prolog_goal import pg_atom, pg_int, pg_var, pg_call, pg_list
    # Params currently unused at codegen; pass empty Prolog list.
    call = pg_call('generate_kernel_pool', [
        pg_atom(op_kind),
        pg_int(dim),
        pg_atom(pool_kind),
        pg_list([]),
        pg_var('K'),
    ])
    return _run_swipl_emit(_construct_goal(call))


# ════════════════════════════════════════════════════════════════════════
# Family 4: Convolutions (im2col)
# ════════════════════════════════════════════════════════════════════════
#
# Mirrors generate_kernel_im2col/4(+OpKind, +Dim, +Mode, -K)

def emit_kernel_im2col(op_kind: str, dim: int,
                        mode: str = 'forward') -> str:
    """Emit CUDA for an im2col / col2im kernel.

    op_kind: ggml_conv_1d/2d/3d, ggml_conv_transpose_1d/2d/3d
    dim: 1, 2, or 3
    mode: forward (standard conv) or transpose (conv_transpose / col2im)
    """
    from prolog_goal import pg_atom, pg_int, pg_var, pg_call
    call = pg_call('generate_kernel_im2col', [
        pg_atom(op_kind),
        pg_int(dim),
        pg_atom(mode),
        pg_var('K'),
    ])
    return _run_swipl_emit(_construct_goal(call))


# ════════════════════════════════════════════════════════════════════════
# Family 5: Losses
# ════════════════════════════════════════════════════════════════════════
#
# Mirrors generate_kernel_loss/4(+OpKind, +ReductionMode, +Params, -K)

def emit_kernel_loss(op_kind: str, reduction_mode: str = 'mean') -> str:
    """Emit CUDA for a loss kernel.

    op_kind: ggml_mse_loss, ggml_cross_entropy_loss, ggml_huber_loss,
             ggml_kl_div_loss, ggml_hinge_loss, ggml_triplet_margin_loss
    reduction_mode: mean or sum
    """
    from prolog_goal import pg_atom, pg_var, pg_call, pg_list
    call = pg_call('generate_kernel_loss', [
        pg_atom(op_kind),
        pg_atom(reduction_mode),
        pg_list([]),
        pg_var('K'),
    ])
    return _run_swipl_emit(_construct_goal(call))


def emit_kernel_activation(op_kind: str) -> str:
    """Emit CUDA for an elementwise unary activation kernel.

    Activations are generated from activation_expr/3 facts in
    lib/kernel_templates_llama.pl. Each fact defines the per-element
    expression; unary_activation_kernel/2 wraps it in the standard
    kernel skeleton.

    op_kind: one of k_silu, k_sigmoid, k_relu, k_gelu, k_tanh.

    Signature: __global__ void <op_kind>(const float * __restrict__ in,
                                         float * __restrict__ out,
                                         int n)

    Body uses math.h functions (expf/erff/tanhf/fmaxf) — the include
    is automatically added by the goal builder.

    Uses the AST-based goal builder (lib/prolog_goal.py) — first call
    site migrated from the legacy string form. The op_kind is wrapped
    as a Prolog atom; K is a Prolog variable to bind the kernel AST.
    """
    from prolog_goal import pg_atom, pg_var, pg_call
    call = pg_call('unary_activation_kernel', [pg_atom(op_kind), pg_var('K')])
    return _run_swipl_emit(_construct_goal(
        call,
        module='lib/kernel_templates_llama',
        includes=('cuda_runtime.h', 'math.h'),
    ))


# ════════════════════════════════════════════════════════════════════════
# Convenience: extract the __global__ function name from emitted source
# ════════════════════════════════════════════════════════════════════════

def emit_reduction_launcher(kernel_name: str) -> str:
    """Emit the extern "C" host launcher for a reduction-family kernel.

    Reductions have the signature:
        __global__ void <kernel_name>(const float* X, float* Y, int N, int outer)

    The launcher signature is:
        extern "C" int launch_<kernel_name>(
            const float* h_X, float* h_Y, int N, int outer)

    Host-callable. Allocates device memory, copies input, launches kernel,
    synchronizes, copies output, frees device memory, returns status.

    Status codes:
        0: success
        1: cudaMalloc input failed
        2: cudaMalloc output failed
        3: cudaMemcpy host->device failed
        4: kernel launch failed (cudaGetLastError)
        5: cudaDeviceSynchronize failed
        6: cudaMemcpy device->host failed

    Args:
        kernel_name: the __global__ kernel's symbol
            (typically obtained via extract_kernel_name)

    Returns:
        CUDA source for the launcher. Concatenate after the kernel source
        before compilation; both compile into the same .so.
    """
    return (
        '\n'
        f'extern "C" int launch_{kernel_name}(\n'
        '        const float* h_X, float* h_Y,\n'
        '        int N, int outer) {\n'
        '    float *d_X = nullptr, *d_Y = nullptr;\n'
        '    size_t size_X = (size_t)outer * (size_t)N * sizeof(float);\n'
        '    size_t size_Y = (size_t)outer * sizeof(float);\n'
        '    cudaError_t err;\n'
        '\n'
        '    err = cudaMalloc((void**)&d_X, size_X);\n'
        '    if (err != cudaSuccess) return 1;\n'
        '    err = cudaMalloc((void**)&d_Y, size_Y);\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); return 2; }\n'
        '\n'
        '    err = cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 3; }\n'
        '\n'
        '    int block = 256;\n'
        '    int grid = (outer + block - 1) / block;\n'
        f'    {kernel_name}<<<grid, block>>>(d_X, d_Y, N, outer);\n'
        '    err = cudaGetLastError();\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 4; }\n'
        '\n'
        '    err = cudaDeviceSynchronize();\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 5; }\n'
        '\n'
        '    err = cudaMemcpy(h_Y, d_Y, size_Y, cudaMemcpyDeviceToHost);\n'
        '    cudaFree(d_X);\n'
        '    cudaFree(d_Y);\n'
        '    if (err != cudaSuccess) return 6;\n'
        '\n'
        '    return 0;\n'
        '}\n'
    )


def emit_activation_launcher(kernel_name: str) -> str:
    """Emit the extern "C" host launcher for an elementwise activation kernel.

    Activations have the signature:
        __global__ void <kernel_name>(const float* in, float* out, int n)

    The launcher signature is:
        extern "C" int launch_<kernel_name>(
            const float* h_X, float* h_Y, int N)

    Simpler than the reduction launcher — no `outer` dimension, just
    a 1D input and 1D output of the same length. Per mavchin's
    description (intercom 10:27 UTC 2026-05-17): "one thread per
    element, output[i] = f(input[i]). No shared memory, no reduction,
    no atomics."

    Status codes mirror emit_reduction_launcher (1=cudaMalloc input,
    2=cudaMalloc output, 3=Memcpy H->D, 4=launch, 5=sync, 6=Memcpy D->H).

    Args:
        kernel_name: the __global__ kernel's symbol (e.g., 'k_silu')

    Returns:
        CUDA source for the launcher. Concatenate after the kernel
        source before compilation; both compile into the same .so.
    """
    return (
        '\n'
        f'extern "C" int launch_{kernel_name}(\n'
        '        const float* h_X, float* h_Y, int N) {\n'
        '    float *d_X = nullptr, *d_Y = nullptr;\n'
        '    size_t size_buf = (size_t)N * sizeof(float);\n'
        '    cudaError_t err;\n'
        '\n'
        '    err = cudaMalloc((void**)&d_X, size_buf);\n'
        '    if (err != cudaSuccess) return 1;\n'
        '    err = cudaMalloc((void**)&d_Y, size_buf);\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); return 2; }\n'
        '\n'
        '    err = cudaMemcpy(d_X, h_X, size_buf, cudaMemcpyHostToDevice);\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 3; }\n'
        '\n'
        '    int block = 256;\n'
        '    int grid = (N + block - 1) / block;\n'
        f'    {kernel_name}<<<grid, block>>>(d_X, d_Y, N);\n'
        '    err = cudaGetLastError();\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 4; }\n'
        '\n'
        '    err = cudaDeviceSynchronize();\n'
        '    if (err != cudaSuccess) { cudaFree(d_X); cudaFree(d_Y); return 5; }\n'
        '\n'
        '    err = cudaMemcpy(h_Y, d_Y, size_buf, cudaMemcpyDeviceToHost);\n'
        '    cudaFree(d_X);\n'
        '    cudaFree(d_Y);\n'
        '    if (err != cudaSuccess) return 6;\n'
        '\n'
        '    return 0;\n'
        '}\n'
    )


def emit_kernel_with_launcher(family: str, op_kind: str, *args) -> tuple:
    """Emit a kernel plus its host-callable launcher.

    The launcher is an `extern "C"` host function that wraps a __global__
    kernel in a ctypes-callable interface. The Python side calls
    `launch_<kernel_name>` with host pointers + dimensions; the launcher
    handles device memory and kernel launch internally.

    Args:
        family: one of 'reduction', 'activation', 'norm', 'pool', 'im2col', 'loss'
        op_kind: family-specific op name
            - reduction: 'ggml_sum_rows', 'ggml_mean', 'ggml_max', etc.
            - activation: 'k_silu', 'k_sigmoid', 'k_relu', 'k_gelu', 'k_tanh'
        *args: family-specific extras

    Returns:
        (combined_source, kernel_name, launcher_name)
        - combined_source: kernel + launcher concatenated, ready for
          compile_kernel_to_so
        - kernel_name: __global__ symbol (for diagnostics)
        - launcher_name: extern "C" symbol to dlsym after .so load
          (typically 'launch_' + kernel_name)
    """
    if family == 'reduction':
        kernel_src = emit_kernel_reduction(op_kind, *args)
        kernel_name = extract_kernel_name(kernel_src)
        if kernel_name is None:
            raise KernelEmitError(
                f"could not extract kernel name from emitted source for "
                f"family={family!r} op_kind={op_kind!r}. Source preview:\n"
                f"{kernel_src[:200]}"
            )
        launcher_src = emit_reduction_launcher(kernel_name)
        combined = kernel_src + launcher_src
        return combined, kernel_name, f'launch_{kernel_name}'
    elif family == 'activation':
        kernel_src = emit_kernel_activation(op_kind, *args)
        kernel_name = extract_kernel_name(kernel_src)
        if kernel_name is None:
            raise KernelEmitError(
                f"could not extract kernel name from emitted source for "
                f"family={family!r} op_kind={op_kind!r}. Source preview:\n"
                f"{kernel_src[:200]}"
            )
        launcher_src = emit_activation_launcher(kernel_name)
        combined = kernel_src + launcher_src
        return combined, kernel_name, f'launch_{kernel_name}'
    elif family in ('norm', 'pool', 'im2col', 'loss'):
        raise NotImplementedError(
            f"launcher template for family={family!r} not yet implemented. "
            f"Reductions only for 1.b.iii; other families extend the matrix "
            f"harness as needed."
        )
    else:
        raise ValueError(f"unknown family: {family!r}")


def extract_kernel_name(cuda_source: str) -> Optional[str]:
    """Parse the kernel function name from emitted CUDA.

    Looks for `__global__ void <name>(`. Returns name or None if not found.
    Used by Scope C-Extended-B to know which symbol to dlsym after .so load.
    """
    import re
    match = re.search(r'__global__\s+void\s+(\w+)\s*\(', cuda_source)
    if match:
        return match.group(1)
    return None


# ════════════════════════════════════════════════════════════════════════
# Convenience: bundle ALL kernels into a single source for batch compile
# ════════════════════════════════════════════════════════════════════════

def emit_all_l1_kernels() -> dict:
    """Emit CUDA for all L1 kernel variants Scope C currently covers.

    Returns a dict mapping kernel_name → cuda_source.
    """
    kernels = {}

    # Reductions
    for op in ['ggml_sum_rows', 'ggml_mean', 'ggml_max', 'ggml_min',
               'ggml_argmax', 'ggml_argmin', 'ggml_cumsum', 'ggml_cumprod']:
        src = emit_kernel_reduction(op)
        name = extract_kernel_name(src)
        if name:
            kernels[name] = src

    # Norms (both affine modes)
    for op in ['ggml_norm', 'ggml_l2_norm', 'ggml_rms_norm', 'ggml_group_norm']:
        for affine in [False, True]:
            src = emit_kernel_norm(op, affine=affine)
            name = extract_kernel_name(src)
            if name:
                # Disambiguate affine vs plain in dict key
                suffix = '_affine' if affine else '_plain'
                kernels[name + suffix] = src

    # Pools (2D only currently has full body)
    for kind in ['max', 'avg']:
        src = emit_kernel_pool('ggml_pool_2d', 2, kind)
        name = extract_kernel_name(src)
        if name:
            kernels[name] = src

    # Conv (2D forward only has full body)
    src = emit_kernel_im2col('ggml_conv_2d', 2, 'forward')
    name = extract_kernel_name(src)
    if name:
        kernels[name] = src

    # Losses
    for op in ['ggml_mse_loss', 'ggml_cross_entropy_loss',
               'ggml_huber_loss', 'ggml_kl_div_loss',
               'ggml_hinge_loss', 'ggml_triplet_margin_loss']:
        src = emit_kernel_loss(op)
        name = extract_kernel_name(src)
        if name:
            kernels[name] = src

    return kernels


if __name__ == '__main__':
    # Smoke test: emit one kernel from each family
    print("=== kernel_emit_bridge.py smoke test ===\n")

    print("Family 1 (Reduction): emit_kernel_reduction('ggml_sum_rows')")
    src = emit_kernel_reduction('ggml_sum_rows')
    name = extract_kernel_name(src)
    print(f"  Kernel name: {name}")
    print(f"  Source length: {len(src)} chars")
    print(f"  First line: {src.splitlines()[0]}")
    print()

    print("Family 2 (Norm): emit_kernel_norm('ggml_rms_norm', affine=True)")
    src = emit_kernel_norm('ggml_rms_norm', affine=True)
    name = extract_kernel_name(src)
    print(f"  Kernel name: {name}")
    print(f"  Source length: {len(src)} chars")
    print()

    print("Family 5 (Loss): emit_kernel_loss('ggml_mse_loss', 'mean')")
    src = emit_kernel_loss('ggml_mse_loss', 'mean')
    name = extract_kernel_name(src)
    print(f"  Kernel name: {name}")
    print(f"  Source length: {len(src)} chars")
    print()

    print("Bundle: emit_all_l1_kernels()")
    bundle = emit_all_l1_kernels()
    print(f"  Total kernels: {len(bundle)}")
    print(f"  Keys: {sorted(bundle.keys())}")
