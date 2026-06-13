# BPD Makefile — substrate verification targets
#
# Per Heath's plan 8d65ba1c (Tier 2 verification, subtask 2-inf-g).
# The bit_identical target is the substantive announcement-ready
# verification artifact: anyone can clone the repo and run
# `make -C bpd bit_identical` to verify the substrate-design claims.
#
# Currently runs on the Tesla P4 enclave (sm_61). Future: cross-architecture
# verification.

ENCLAVE := metayen-enclave
NVIDIA_LIB := /nix/store/a6kbivfsa0rscf11l4373v80c5c6l6na-nvidia-x11-570.153.02-6.12.42/lib
TIER2_DIR := tests/tier2

.PHONY: bit_identical bit_identical_local clean_tier2 help

help:
	@echo "BPD Verification Targets"
	@echo "========================"
	@echo "  bit_identical        Run the full KernelBench L1 bit-identical sweep"
	@echo "                       on the enclave. Compares substrate output to"
	@echo "                       PyTorch references; detects stubs; reports"
	@echo "                       per-kernel status."
	@echo "  bit_identical_local  Same, but assumes you're already on a CUDA host"
	@echo "                       with PyTorch + nvcc available."
	@echo "  clean_tier2          Remove tier2 build artifacts."

# Sync tier2 files to enclave, run sweep, capture output.
bit_identical:
	@echo "=== BPD Tier 2 bit-identical verification ==="
	@echo "Target: Tesla P4 enclave (sm_61)"
	@echo ""
	@# Sync the harness scripts
	@scp -q $(TIER2_DIR)/bit_identical_v1.py $(ENCLAVE):/tmp/
	@scp -q $(TIER2_DIR)/bit_identical_v2.py $(ENCLAVE):/tmp/
	@# Make sure substrate emit exists on enclave (re-run Tier 1 harness if needed)
	@ssh $(ENCLAVE) 'test -f /tmp/l1_cuda_validation/reduce_sum_rows.cu || { \
	    echo "Warning: substrate emit not found on enclave. Re-run Tier 1 harness first."; exit 1; }'
	@# Run the sweep
	@ssh $(ENCLAVE) 'NIXPKGS_ALLOW_UNFREE=1 nix-shell -p python3 \
	    python3Packages.pytorch python3Packages.numpy \
	    cudaPackages.cuda_nvcc cudaPackages.cuda_cudart --impure --run \
	    "cd /tmp && LD_LIBRARY_PATH=$(NVIDIA_LIB):\$$LD_LIBRARY_PATH \
	    python3 bit_identical_v2.py"'

bit_identical_local:
	@LD_LIBRARY_PATH=$(NVIDIA_LIB):$$LD_LIBRARY_PATH \
	    python3 $(TIER2_DIR)/bit_identical_v2.py

clean_tier2:
	@ssh $(ENCLAVE) 'rm -rf /tmp/tier2_build /tmp/bit_identical_v*.py' 2>/dev/null || true
	@echo "Cleaned tier2 build artifacts on enclave."

# SASS audit — disassemble substrate-emitted kernels, characterize instructions.
# Per Heath's interjection 2026-05-19: SASS reveals what kernels do at silicon level
# beyond what source code shows. Bit-identity vs divergence is empirically grounded
# in instruction mix (FADD = sequential reduction, FMNMX = order-independent).
.PHONY: sass_audit

sass_audit:
	@echo "=== BPD SASS Audit ==="
	@scp -q $(TIER2_DIR)/sass_audit.py $(ENCLAVE):/tmp/
	@ssh $(ENCLAVE) 'NIXPKGS_ALLOW_UNFREE=1 nix-shell -p python3 \
	    cudaPackages.cuda_nvcc cudaPackages.cuda_cuobjdump cudaPackages.cuda_nvdisasm --impure --run \
	    "python3 /tmp/sass_audit.py"'
