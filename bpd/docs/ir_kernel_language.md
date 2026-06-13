# The IR Kernel Language — subsuming the taxonomy of existing kernel softwares

(Iyun, 2026-05-29, Heath.) The vision: FLUENCY in LLVM IR generation so we can MATCH every reference
implementation of GPU or CPU kernels — and because our parameterized IR generator is a SUPERIOR
LANGUAGE for handling these programs, it SUBSUMES the taxonomy of existing softwares.

## THE CORE IDEA: existing softwares are POINTS in our parameter space
Each existing kernel software made specific choices and is therefore one expressible POINT:
- ggml/llama.cpp  = (quant=q8_0/q4_K, accum=i32, scale=fp16, reduce=avx2_fma_tree, target=cpu_avx2)
- cuBLAS sgemm    = (dtype=f32, accum=f32, reduce=warp_shuffle, target=gpu_tensor_core)
- PyTorch eager   = (dispatch=cuDNN/MKL, opaque)
- llama.cpp CUDA  = (quant=q8_0, dp4a int8, target=gpu_sm_61)
Our GENERATOR is the SPACE ITSELF. Each software = one KernelSpec. We don't reimplement them; we
EXPRESS them as data, then generate, MEASURE (vs that reference), optimize, and RETARGET.

## WHY THIS IS A SUPERIOR LANGUAGE (subsumption, not just another library)
- ggml: C + hand-written intrinsics, ONE ISA at a time, hand-tuned. Cannot retarget without rewriting.
- PyTorch: dispatches to opaque vendor libs (cuBLAS/cuDNN), vendor-locked.
- OURS: parameterized LLVM IR — ABOVE all of them. It can EXPRESS what each is (as parameters),
  compile to ANY target (LLVM backends: x86, NVPTX, AMDGPU, ARM, RISC-V), and generate points they
  CANNOT express (optimizations past the reference, verified against the reference fixture).
Subsumption = the meta-language in which all kernel libraries are expressible points + a measurement
discipline that proves match/exceed/retarget.

## THE FOUR OPERATIONS the language must support FLUENTLY
1. MATCH:    emit IR for a reference's point -> compile -> execute -> MEASURE 0 ULP vs that reference.
2. SUBSUME:  name the reference as a KernelSpec in our space (it becomes a point, callable + composable).
3. EXCEED:   generate NEIGHBORING points (wider SIMD, butterfly reduce, fused scales) -> re-MEASURE
             vs the reference fixture (stay 0 ULP OR a declared improvement). Optimize FROM the anchor.
4. RETARGET: same logical KernelSpec, target=cpu_avx2 | gpu_sm_61 | amdgpu | riscv_v -> one source,
             every backend, each re-MEASURED on its platform.

## FLUENCY = a rich enough PRIMITIVE VOCABULARY to compose any kernel
The generator (llvmlite-backed, PROVEN: ir_q8dot.py = 0 ULP measured) needs fluent primitives:
- quantize(dtype)         : f32->q8_0/q4_K/q8_K... (the vec_dot_type transforms)
- dequantize(dtype)
- block_dot(wq,aq,accum)  : the int/fp dot over a block (PROVEN for q8_0)
- reduce(order)           : sequential | avx2_fma_tree | warp_shuffle | butterfly
- scale(fp_order)         : the exact fp scale/accumulate op order
- layout(transform)       : transpose/permute/tile (composes layout_algebra — already built)
- matmul(blocks,rows)     : compose block_dot over a tensor (PROVEN: ir_q8_matmul.py)
- rope/softmax/rmsnorm    : the non-matmul Mistral ops
Each primitive emits IR parameterized by target; each is MEASURED against the critical-path reference
(llama.cpp) with numpy/pytorch as corroborating cross-checks.

## THE MEASUREMENT DISCIPLINE (what makes subsumption REAL, not asserted)
A reference is SUBSUMED only when our generated IR for its point EXECUTES and MEASURES 0 ULP vs it.
- critical-path reference = llama.cpp (market = Ollama). green-defining.
- corroborating references = numpy, pytorch, HF (confirm math, catch lift errors).
Never "subsumed by construction" — subsumed by EXECUTED COVERAGE. The divergence dashboard tracks it.

## STATUS (proven tonight)
- llvmlite emit -> JIT -> execute -> MEASURE pipeline: PROVEN (ir_q8dot.py block-dot = 0 ULP).
- full matmul composed + executed: 6.4e-5 vs llama.cpp (math right via roundf(x*id); exact-0 needs
  compiled IR running hardware FMA/fp16 — the point of the IR path).
- existing parameter spaces: gemm_pattern/6, QuantDotParams (seeds of the meta-language).
- toolchain: clang19, llvmlite0.44, NVPTX backend (CPU + P4 GPU reachable).

## NEXT toward fluency
1. Unify gemm_pattern + QuantDotParams + layout_algebra into ONE KernelSpec language (llvmlite-backed).
2. Make MATCH fluent: given a reference (e.g. an eval-callback op), emit+compile+execute+measure.
3. Build the primitive vocabulary (reduce/scale/quantize as composable IR emitters).
4. Retarget proof: same KernelSpec -> CPU AND P4 GPU (NVPTX), both measured.
