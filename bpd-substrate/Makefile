# bpd-substrate — Makefile
#
# THE MENU (10 high-level goals; fine-grain via FOCUS=name):
#
#   make help        Show this menu.
#   make build       Compile all kernel .so artifacts.
#   make test        Run everything: lint + correctness + verify + tier1 + smoke.
#   make lint        Prolog module sanity check (zero-warning policy).
#   make correctness Wilkinson backward-error mathematical correctness harness.
#   make verify      Bit-identity sweeps (CPU + cuBLAS + YOLO).
#   make tier1       KernelBench L1 structural + nvcc compile validation.
#   make smoke       Fresh-clone smoke tests (mavchin's Rule 2 harness).
#   make bench       Performance benchmarks.
#   make clean       Remove build/.
#
# FINE-GRAIN via FOCUS=name (selects a single test or category):
#
#   make verify FOCUS=blas              Just BLAS L1 verification.
#   make verify FOCUS=cpu               Just CPU bit-identity sweep.
#   make verify FOCUS=cublas            Just cuBLAS 0-ULP bit-identity.
#   make verify FOCUS=yolo              All YOLO harnesses (per-stage + layer2).
#   make verify FOCUS=yolo-per-stage    Just verify_yolo_per_stage.py.
#   make verify FOCUS=yolo-layer2       Just verify_yolo_layer2_c3.py.
#   make verify FOCUS=layer2-primitives Just residual_add + concat verification.
#   make verify FOCUS=upsample          Just verify_upsample.py.
#   make verify FOCUS=opmath            Just opmath_precision TDD harness.
#   make verify FOCUS=kernelbench-l1    Full 28-family Tier 2 sweep (slow).
#   make verify FOCUS=kernelbench-l1-cpu Stanford L1 CPU bit-identity sweep (100 problems).
#   make verify FOCUS=gemm-sweep        4328 GEMM kernels vs cblas_sgemm bit-identity sweep.
#   make verify FOCUS=gemm-perf         GFLOPS measurement across BIT_IDENTICAL GEMM patterns.
#   make verify FOCUS=cascade-sweep     160 cascade reduction kernels vs torch.sum.
#
#   make test FOCUS=lint                Just lint.
#   make test FOCUS=correctness         Just correctness.
#   make test FOCUS=spec                Just spec_conformance.pl (Rule 1).
#   make test FOCUS=stage-boundary      Just stage_boundary_verification.py (Rule 3).
#
#   make bench FOCUS=fusion             Just bench_fusion.py.
#   make bench FOCUS=perftest           Just bench/perftest.py (A/B/C).
#
# Configuration (override on the command line):
#   NVCC_ARCH        GPU SM target (default sm_86 = Ampere RTX 30xx).
#                    Set to sm_61 for Pascal, sm_89 for Ada.
#   BUILD_DIR        Output directory for .cu/.so artifacts (default build).
#   SWIPL            Path to swipl (default: swipl on PATH).
#   NVCC             Path to nvcc (default: nvcc on PATH).
#   PYTHON           Python interpreter (default: python3).
#   CPU_FP_MODE      CPU floating-point mode: strict (default), fma, native, or mkl.
#                    mkl: targets bit-identity with PyTorch builds that link Intel MKL.

BUILD_DIR ?= build
NVCC_ARCH ?= sm_86
SWIPL     ?= swipl
NVCC      ?= nvcc
PYTHON    ?= python3
FOCUS     ?=
CPU_FP_MODE ?= strict

NVCC_FLAGS = -arch=$(NVCC_ARCH) -O2 -shared -Xcompiler -fPIC

CPU_FP_strict = -O2
CPU_FP_fma    = -O2 -mfma -ffp-contract=on
CPU_FP_native = -O2 -march=native
# mkl: AVX2+FMA, BPD_MKL_PATH=1 enables SVML-polynomial transcendentals and
# AVX2 GEMV/affine-apply paths that match Intel MKL's PyTorch backend.
CPU_FP_mkl    = -O2 -mavx2 -mfma -ffp-contract=on -DBPD_MKL_PATH=1
CPU_FPFLAGS   = $(CPU_FP_$(CPU_FP_MODE))

GENERATORS = blas fused llama
SOS        = $(addprefix $(BUILD_DIR)/, $(addsuffix _kernels.so, $(GENERATORS)))

.PHONY: help build test verify bench lint correctness tier1 smoke clean
.PHONY: $(GENERATORS) blas fused llama
.PHONY: kernelbench_l1

# Default target: show the menu so first-time visitors see what exists.
.DEFAULT_GOAL := help

help:
	@awk '/^# THE MENU/,/^# Configuration/' Makefile | sed 's/^# //; s/^#$$//'

# ─── Build ────────────────────────────────────────────────────────────────

build: $(SOS)

blas:  $(BUILD_DIR)/blas_kernels.so
fused: $(BUILD_DIR)/fused_kernels.so
llama: $(BUILD_DIR)/llama_kernels.so

# Prolog → .cu
$(BUILD_DIR)/%_kernels.cu: generators/generate_%_kernels.pl lib/*.pl
	@mkdir -p $(@D)
	@echo "[swipl] $@"
	@$(SWIPL) $< > $@

# .cu → .so
$(BUILD_DIR)/%_kernels.so: $(BUILD_DIR)/%_kernels.cu
	@echo "[nvcc -arch=$(NVCC_ARCH)] $@"
	@$(NVCC) $(NVCC_FLAGS) -o $@ $<

# CPU substrate library (no GPU required)
$(BUILD_DIR)/bpd_cpu.so: bench/bpd_cpu.c bench/bpd_gemm_q8_0_cpu.c
	@mkdir -p $(@D)
	@echo "[gcc $(CPU_FP_MODE)] $@"
	@gcc $(CPU_FPFLAGS) -shared -fPIC -o $@ $< -lm

# Matmul shared library for cuBLAS bit-identity test
$(BUILD_DIR)/bpd_mm.so: lib/kernel_templates_blas.pl
	@mkdir -p $(@D)
	@echo "[sgemm] $@"
	@$(SWIPL) -g 'use_module("lib/kernel_templates_blas"), halt' 2>/dev/null
	@$(NVCC) -arch=$(NVCC_ARCH) -O3 -shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets -o $@ bench/mm_shared.cu

# ─── Tests ────────────────────────────────────────────────────────────────

# `make test` runs everything. `make test FOCUS=name` runs one category.
# This is the "does the whole project pass?" target.
test:
ifeq ($(FOCUS),)
	@$(MAKE) -s lint
	@$(MAKE) -s correctness
	@$(MAKE) -s verify
	@$(MAKE) -s tier1
	@$(MAKE) -s smoke
	@echo
	@echo "[test] ALL PASS — substrate is healthy."
else ifeq ($(FOCUS),lint)
	@$(MAKE) -s lint
else ifeq ($(FOCUS),correctness)
	@$(MAKE) -s correctness
else ifeq ($(FOCUS),spec)
	@echo "[test FOCUS=spec] running spec-conformance tests (Rule 1)..."
	@cd tests && $(SWIPL) -q -g 'consult(test_spec_conformance), run_tests' -t 'halt(1)'
else ifeq ($(FOCUS),stage-boundary)
	@echo "[test FOCUS=stage-boundary] running stage-boundary verification (Rule 3)..."
	@$(PYTHON) tests/test_stage_boundary_verification.py
else
	@echo "Unknown FOCUS=$(FOCUS). Try: lint, correctness, spec, stage-boundary."
	@echo "Or run 'make help' for the full menu."
	@exit 1
endif

lint:
	@echo "[lint] checking Prolog modules..."
	@$(SWIPL) --on-warning=status -g 'use_module("lib/c_ast"), use_module("lib/kernel_templates_blas"), use_module("lib/kernel_templates"), use_module("lib/auto_fuser"), use_module("lib/epilogue_generator"), use_module("lib/fusion_optimizer"), use_module("lib/matmul_optimizer"), use_module("lib/matmul_cycle_model"), use_module("lib/graph_complexity"), halt(0)' || (echo "FAIL: Prolog warnings detected." && exit 1)
	@echo "[lint] all clean — zero warnings."

correctness: $(BUILD_DIR)/bpd_cpu.so
	@echo "[correctness] Wilkinson backward-error mathematical correctness harness..."
	@$(PYTHON) bench/test_correctness.py

# Tier 1: structural + compile validation of all 28 KernelBench L1 families.
tier1:
	@echo "[tier1] running KernelBench L1 structural + compile validation..."
	@cd tests && $(SWIPL) -q -g 'consult(test_kernelbench_l1_structure), run_tests' -t 'halt(1)'
	@cd tests && $(SWIPL) -q -g 'consult(test_kernelbench_l1_cuda), run_tests' -t 'halt(1)'

# Smoke tests: mavchin's Rule 2 harness (fresh-clone non-trivial output).
smoke:
	@echo "[smoke] running fresh-clone smoke tests (Rule 2)..."
	@bash tests/test_fresh_clone_smoke.sh

# ─── Verify: bit-identity sweeps ──────────────────────────────────────────
#
# `make verify` runs the standard sweep (CPU + BLAS + YOLO + opmath).
# `make verify FOCUS=name` runs one specific harness.

verify: $(BUILD_DIR)/bpd_cpu.so
ifeq ($(FOCUS),)
	@$(MAKE) -s verify FOCUS=cpu
	@$(MAKE) -s verify FOCUS=blas
	@$(MAKE) -s verify FOCUS=opmath
	@$(MAKE) -s verify FOCUS=layer2-primitives
	@$(MAKE) -s verify FOCUS=yolo
else ifeq ($(FOCUS),cpu)
	@echo "[verify cpu] BPD CPU vs PyTorch CPU bit-identity sweep..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/bit_identical_universal.py
else ifeq ($(FOCUS),cublas)
	@$(MAKE) -s $(BUILD_DIR)/bpd_mm.so
	@echo "[verify cublas] BPD vs cuBLAS 0-ULP bit-identity..."
	@BPD_MM_SO=$(abspath $(BUILD_DIR)/bpd_mm.so) $(PYTHON) bench/bit_identical.py
else ifeq ($(FOCUS),blas)
	@$(MAKE) -s $(BUILD_DIR)/blas_kernels.so
	@echo "[verify blas] BLAS L1 verification (ColonistOne PR #1)..."
	@BPD_BUILD_DIR=$(abspath $(BUILD_DIR)) $(PYTHON) bench/verify_blas.py
else ifeq ($(FOCUS),opmath)
	@echo "[verify opmath] opmath_precision invariance TDD harness..."
	@$(PYTHON) bench/test_opmath_precision_invariance.py
else ifeq ($(FOCUS),layer2-primitives)
	@echo "[verify layer2-primitives] residual_add + concat bit-identity..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/verify_layer2_primitives.py
else ifeq ($(FOCUS),upsample)
	@echo "[verify upsample] upsample bit-identity..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/verify_upsample.py
else ifeq ($(FOCUS),yolo)
	@$(MAKE) -s verify FOCUS=yolo-per-stage
	@$(MAKE) -s verify FOCUS=yolo-layer2
else ifeq ($(FOCUS),yolo-per-stage)
	@echo "[verify yolo-per-stage] YOLO Layer 0+1 per-stage bit-identity..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/verify_yolo_per_stage.py
else ifeq ($(FOCUS),yolo-layer2)
	@echo "[verify yolo-layer2] YOLO Layer 2 (C3) end-to-end bit-identity..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/verify_yolo_layer2_c3.py /tmp/yolov5n.pt
else ifeq ($(FOCUS),kernelbench-l1)
	@$(MAKE) -s tier1
	@echo "[verify kernelbench-l1] 28-family Tier 2 sweep (slow)..."
	@$(PYTHON) bench/tier2/bit_identical_v1.py
	@$(PYTHON) bench/tier2/bit_identical_v2.py
	@$(PYTHON) bench/tier2/bit_identical_v3.py
else ifeq ($(FOCUS),kernelbench-l1-cpu)
	@echo "[verify kernelbench-l1-cpu] Stanford L1 CPU bit-identity sweep (100 problems)..."
	@BPD_CPU_SO=$(abspath $(BUILD_DIR)/bpd_cpu.so) $(PYTHON) bench/verify_kernelbench_l1_cpu.py
else ifeq ($(FOCUS),gemm-sweep)
	@echo "[verify gemm-sweep] Generating + compiling 4328 GEMM kernels..."
	@$(PYTHON) bench/generate_gemm_kernels.py > bench/gemm_kernels_generated.c
	@gcc -O2 -shared -fPIC -o $(BUILD_DIR)/gemm.so bench/gemm_kernels_generated.c -lm
	@echo "[verify gemm-sweep] Running bit-identity sweep vs cblas_sgemm..."
	@GEMM_SO=$(abspath $(BUILD_DIR)/gemm.so) $(PYTHON) bench/verify_gemm_sweep.py
else ifeq ($(FOCUS),gemm-perf)
	@echo "[verify gemm-perf] GFLOPS measurement across BIT_IDENTICAL GEMM patterns..."
	@test -f $(BUILD_DIR)/gemm.so || ( $(PYTHON) bench/generate_gemm_kernels.py > bench/gemm_kernels_generated.c && gcc -O2 -shared -fPIC -o $(BUILD_DIR)/gemm.so bench/gemm_kernels_generated.c -lm )
	@GEMM_SO=$(abspath $(BUILD_DIR)/gemm.so) $(PYTHON) bench/perf_gemm_sweep.py
else ifeq ($(FOCUS),cascade-sweep)
	@echo "[verify cascade-sweep] Generating + compiling 160 cascade reduction kernels..."
	@$(PYTHON) bench/generate_cascade_kernels.py > bench/cascade_kernels_generated.c
	@gcc -O2 -shared -fPIC -o $(BUILD_DIR)/cascade.so bench/cascade_kernels_generated.c -lm
	@echo "[verify cascade-sweep] Running bit-identity sweep vs PyTorch torch.sum..."
	@CASCADE_SO=$(abspath $(BUILD_DIR)/cascade.so) $(PYTHON) bench/verify_cascade_sweep.py
else
	@echo "Unknown FOCUS=$(FOCUS). Try one of:"
	@echo "  cpu, cublas, blas, opmath,"
	@echo "  layer2-primitives, upsample,"
	@echo "  yolo, yolo-per-stage, yolo-layer2, kernelbench-l1, kernelbench-l1-cpu,"
	@echo "  gemm-sweep, gemm-perf, cascade-sweep."
	@exit 1
endif

# Alias for back-compat with existing docs / contributors.
kernelbench_l1:
	@$(MAKE) -s verify FOCUS=kernelbench-l1

# ─── Benchmarks ───────────────────────────────────────────────────────────
#
# `make bench` runs the standard set. `make bench FOCUS=name` runs one.

bench: $(BUILD_DIR)/blas_kernels.so
ifeq ($(FOCUS),)
	@$(MAKE) -s bench FOCUS=fusion
else ifeq ($(FOCUS),fusion)
	@echo "[bench fusion] BPD-fused vs PyTorch-unfused..."
	@BPD_BUILD_DIR=$(abspath $(BUILD_DIR)) $(PYTHON) bench/bench_fusion.py
else ifeq ($(FOCUS),perftest)
	@echo "[bench perftest] airtight A/B/C inside PyTorch runtime..."
	@$(PYTHON) bench/perftest.py
else
	@echo "Unknown FOCUS=$(FOCUS). Try: fusion, perftest."
	@exit 1
endif

# ─── Clean ────────────────────────────────────────────────────────────────

clean:
	@rm -rf $(BUILD_DIR)

# GPU kernel library with host-callable wrappers
$(BUILD_DIR)/bpd_gpu.so: bench/bpd_gpu_kernels.cu
	@mkdir -p $(@D)
	@echo "[nvcc] $@"
	@nvcc -O2 -shared -Xcompiler -fPIC -o $@ $< -arch=$(NVCC_ARCH)
