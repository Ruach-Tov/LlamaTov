# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
toolchain.py — THE single source of CUDA/toolchain environment discovery.

WHY THIS EXISTS (from the unified review, 2026-06-13): the same env lesson was learned
SEPARATELY by production (fact_dispatch.py) and by the tests/sweeps (KernelBench L1 broke
THREE distinct ways on nix split-outputs; gemm/kernelbench sweeps each hardcoded their own
nvcc path). Eight files independently reinvented "where is CUDA" — and the test copies were
wrong. This module is the ONE place that knows, so production and tests learn env lessons
TOGETHER. Defense against the ENVIRONMENT-SHIFT time-constant.

Discovery order (each step falls through to the next, LOUD not silent):
  CUDA root:   $CUDA_HOME -> $CUDA_PATH -> dirname(dirname(realpath(which nvcc))) -> the known
               nix cuda-merged store path -> raise.
  nvcc:        $CUDA_ROOT/bin/nvcc (verified to exist + runnable).
  includes:    $CUDA_ROOT/include (cuda_runtime.h must be present — the bug that bit L1).
  libcuda:     the DRIVER stub at /run/opengl-driver/lib (NOT in the cuda-merged tree — a
               trap that cost a debug cycle), then standard locations.

Public API:
  cuda_root() -> str            the toolkit root (has bin/, include/, lib/)
  nvcc() -> str                 absolute path to a working nvcc
  cuda_include() -> str         the include dir containing cuda_runtime.h
  nvcc_env() -> dict            os.environ + PATH/CPATH/LD_LIBRARY_PATH for compiling+running
  nvcc_compile_cmd(cu, out, arch='sm_61', extra=()) -> list   the canonical compile argv
  libcuda() -> ctypes.CDLL      the driver library (for cuMemAlloc/cuLaunchKernel)
  describe() -> dict            everything discovered (for diagnostics + the Prolog emitter)
  write_prolog_facts(path)      emit toolchain_fact/2 so swipl tests share the SAME discovery
"""
import os, shutil, subprocess, ctypes, json

# The known nix cuda-merged store path — the LAST-RESORT fallback, not the first choice.
_KNOWN_CUDA = "/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8"
# The driver stub lives here on this enclave (NOT in the cuda-merged tree, whose -lcuda is a stub).
_DRIVER_LIB_DIRS = ["/run/opengl-driver/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"]

_cache = {}


def _which_nvcc_root():
    """dirname(dirname(realpath(which nvcc))) — the toolkit root if nvcc is on PATH."""
    p = shutil.which("nvcc")
    if not p:
        return None
    real = os.path.realpath(p)              # follow the symlink (nix wrappers)
    root = os.path.dirname(os.path.dirname(real))
    return root if os.path.isdir(os.path.join(root, "include")) else None


def _is_complete_root(c):
    """A root is COMPLETE only if it has BOTH bin/nvcc AND include/cuda_runtime.h. The nix split-
    output trap: `which nvcc` finds the cuda_nvcc package whose include/ lacks cuda_runtime.h (those
    live in cuda_cudart). A binary-only root is the exact thing that broke L1 — reject it here."""
    return bool(c) and os.path.exists(os.path.join(c, "bin", "nvcc")) \
        and os.path.exists(os.path.join(c, "include", "cuda_runtime.h"))


def cuda_root():
    if "root" in _cache:
        return _cache["root"]
    cands = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_ROOT"),
        _which_nvcc_root(),
        _KNOWN_CUDA,                       # the cuda-MERGED tree: has binary AND headers together
    ]
    # First pass: a COMPLETE root (binary + headers) — what compilation actually needs.
    for c in cands:
        if _is_complete_root(c):
            _cache["root"] = c
            return c
    # No complete root: report what we found so the failure names the split-output trap.
    found = [c for c in cands if c and os.path.exists(os.path.join(c, "bin", "nvcc"))]
    raise RuntimeError(
        "toolchain: no COMPLETE CUDA root (bin/nvcc + include/cuda_runtime.h). "
        f"Found nvcc-only roots (the nix split-output trap): {found}. "
        f"The merged tree {_KNOWN_CUDA} has both; set CUDA_HOME to a complete toolkit.")


def nvcc():
    if "nvcc" in _cache:
        return _cache["nvcc"]
    n = os.path.join(cuda_root(), "bin", "nvcc")
    if not os.path.exists(n):
        raise RuntimeError(f"toolchain: nvcc not at {n}")
    _cache["nvcc"] = n
    return n


def cuda_include():
    if "include" in _cache:
        return _cache["include"]
    inc = os.path.join(cuda_root(), "include")
    # The bug that bit KernelBench L1 THREE times: cuda_runtime.h must actually be here.
    if not os.path.exists(os.path.join(inc, "cuda_runtime.h")):
        raise RuntimeError(
            f"toolchain: cuda_runtime.h NOT in {inc} (nix split-output trap). "
            "This is the exact gap that broke the L1 harness; failing loud.")
    _cache["include"] = inc
    return inc


def _driver_lib_dir():
    for d in _DRIVER_LIB_DIRS:
        if os.path.exists(os.path.join(d, "libcuda.so")) or os.path.exists(os.path.join(d, "libcuda.so.1")):
            return d
    return _DRIVER_LIB_DIRS[0]  # best-effort default (the enclave's known location)


def nvcc_env():
    """os.environ augmented so BOTH nvcc compilation AND runtime (cuLaunch via the driver) work."""
    root = cuda_root()
    drv = _driver_lib_dir()
    return dict(
        os.environ,
        PATH=f"{root}/bin:" + os.environ.get("PATH", ""),
        CPATH=f"{root}/include:" + os.environ.get("CPATH", ""),
        LD_LIBRARY_PATH=f"{drv}:{root}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
    )


def nvcc_compile_cmd(cu, out, arch="sm_61", cubin=True, extra=()):
    """The canonical nvcc argv — with the -I include flag that was the recurring omission."""
    cmd = [nvcc(), f"-arch={arch}", "-O3", f"-I{cuda_include()}"]
    cmd += (["-cubin"] if cubin else ["-c"])
    cmd += list(extra)
    cmd += [cu, "-o", out]
    return cmd


def _devrt_dir():
    """Find the dir holding libcudadevrt.a (needed to LINK full executables, not just compile cubins).
    On nix it's in a separate output; the sweeps each ran a `find /nix/store` for it. Done once here."""
    if "devrt" in _cache:
        return _cache["devrt"]
    # prefer the merged tree's lib, then a store-wide find (cached).
    cand = os.path.join(cuda_root(), "lib", "libcudadevrt.a")
    if os.path.exists(cand):
        _cache["devrt"] = os.path.dirname(cand)
        return _cache["devrt"]
    try:
        r = subprocess.run("find /nix/store -name libcudadevrt.a 2>/dev/null | head -1",
                           shell=True, capture_output=True, text=True, timeout=20)
        p = r.stdout.strip()
        _cache["devrt"] = os.path.dirname(p) if p else os.path.join(cuda_root(), "lib")
    except Exception:
        _cache["devrt"] = os.path.join(cuda_root(), "lib")
    return _cache["devrt"]


def link_lib_dirs():
    """All -L dirs needed to LINK a full CUDA executable: toolkit lib, devrt, driver stub. The
    sweeps each hardcoded their own (some with extra REDIST/STUBS nix paths that drift). This is the
    shared list; callers can append project-specific extras."""
    return [os.path.join(cuda_root(), "lib"), _devrt_dir(), _driver_lib_dir()]


def nvcc_link_cmd(src, out, arch="sm_61", libs=("cuda",), extra_L=(), extra=()):
    """The canonical nvcc argv for building a runnable executable (cudart shared + driver link)."""
    cmd = [nvcc(), "-O3", f"-arch={arch}", "-cudart", "shared", f"-I{cuda_include()}"]
    for d in list(link_lib_dirs()) + list(extra_L):
        cmd.append(f"-L{d}")
    cmd += list(extra)
    cmd += [src]
    for lib in libs:
        cmd.append(f"-l{lib}")
    cmd += ["-o", out]
    return cmd


def libcuda():
    if "libcuda" in _cache:
        return _cache["libcuda"]
    drv = _driver_lib_dir()
    for cand in (os.path.join(drv, "libcuda.so"),
                 os.path.join(drv, "libcuda.so.1"),
                 "libcuda.so", "libcuda.so.1"):
        try:
            lib = ctypes.CDLL(cand)
            _cache["libcuda"] = lib
            return lib
        except OSError:
            continue
    raise RuntimeError(f"toolchain: libcuda.so not loadable from {_DRIVER_LIB_DIRS}")


def describe():
    """Everything discovered — for diagnostics and the Prolog fact emitter."""
    d = {"cuda_root": cuda_root(), "nvcc": nvcc(),
         "cuda_include": cuda_include(), "driver_lib_dir": _driver_lib_dir()}
    try:
        v = subprocess.run([nvcc(), "--version"], capture_output=True, text=True, timeout=10)
        d["nvcc_version"] = (v.stdout.strip().splitlines() or ["?"])[-1]
    except Exception as e:
        d["nvcc_version"] = f"err: {e}"
    return d


def write_prolog_facts(path):
    """Emit toolchain_fact/2 so swipl-driven tests share the SAME discovery (no separate copy)."""
    d = describe()
    with open(path, "w") as f:
        f.write("%% GENERATED by toolchain.py — the single source of CUDA env truth.\n")
        f.write("%% swipl tests: consult this instead of hardcoding /nix paths.\n")
        for k, v in d.items():
            f.write(f"toolchain_fact('{k}', '{v}').\n")
    return path


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
