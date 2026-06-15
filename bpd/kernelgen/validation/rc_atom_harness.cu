// SPDX-License-Identifier: LicenseRef-RTAAL-1.1
// rc_atom_harness.cu — validate the GENERATED kv_direct_recompute against the Python reference K_ref.
// Reads a flat binary fixture (residual, attn_norm_w, Wk, K_ref) and runs the generated CUDA recompute
// on the REAL P4, comparing K_gpu vs K_ref (bounded-ULP gate). This validates the recompute ATOM.
// The generated kernels (k_rms_norm, k_vecmat, kv_direct_recompute) + helpers are #included below.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "residual_cache_gen.cuh"   // the GENERATED kernels + kv_direct_recompute wrapper

static void* slurp(const char* path, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    void* p = malloc(bytes);
    size_t got = fread(p, 1, bytes, f);
    if (got != bytes) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, bytes); exit(2); }
    fclose(f);
    return p;
}

int main(int argc, char** argv) {
    // dims passed on cmdline: embd k_out  ; fixture files are raw float32 dumps
    int embd  = atoi(argv[1]);
    int k_out = atoi(argv[2]);
    float eps = atof(argv[3]);
    float* residual    = (float*)slurp("/tmp/fx_residual.bin",  (size_t)embd*4);
    float* attn_norm_w = (float*)slurp("/tmp/fx_attn_norm.bin", (size_t)embd*4);
    float* Wk          = (float*)slurp("/tmp/fx_wk.bin",        (size_t)embd*k_out*4);
    float* K_ref       = (float*)slurp("/tmp/fx_kref.bin",      (size_t)k_out*4);

    float *d_res, *d_nw, *d_wk, *d_kv, *d_scratch;
    cudaMalloc(&d_res, (size_t)embd*4);
    cudaMalloc(&d_nw,  (size_t)embd*4);
    cudaMalloc(&d_wk,  (size_t)embd*k_out*4);
    cudaMalloc(&d_kv,  (size_t)k_out*4);
    cudaMalloc(&d_scratch, (size_t)embd*4);
    cudaMemcpy(d_res, residual,    (size_t)embd*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nw,  attn_norm_w, (size_t)embd*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wk,  Wk,          (size_t)embd*k_out*4, cudaMemcpyHostToDevice);

    // THE GENERATED ORCHESTRATION: kv_direct_recompute = rms_norm(residual) then proj_w @ normed
    kv_direct_recompute(d_res, d_nw, d_wk, d_kv, d_scratch, embd, k_out, eps);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { fprintf(stderr, "cuda err: %s\n", cudaGetErrorString(e)); return 3; }

    float* K_gpu = (float*)malloc((size_t)k_out*4);
    cudaMemcpy(K_gpu, d_kv, (size_t)k_out*4, cudaMemcpyDeviceToHost);

    int exact=0, maxulp=0; double maxabs=0, sumabs=0;
    for (int i=0;i<k_out;i++) {
        float a=K_ref[i], b=K_gpu[i];
        int32_t ab,bb; __builtin_memcpy(&ab,&a,4); __builtin_memcpy(&bb,&b,4);
        int ulp=abs(ab-bb); if(ulp>maxulp)maxulp=ulp; if(ulp==0)exact++;
        double d=fabs((double)a-(double)b); if(d>maxabs)maxabs=d; sumabs+=d;
    }
    printf("=== kv_direct_recompute ATOM validation (GPU vs Python K_ref, %d elems) ===\n", k_out);
    printf("  K_gpu[0..3] = %.6f %.6f %.6f %.6f\n", K_gpu[0],K_gpu[1],K_gpu[2],K_gpu[3]);
    printf("  K_ref[0..3] = %.6f %.6f %.6f %.6f\n", K_ref[0],K_ref[1],K_ref[2],K_ref[3]);
    printf("  exact=%d/%d  max_ulp=%d  max_abs=%.3e  mean_abs=%.3e\n",
           exact, k_out, maxulp, maxabs, sumabs/k_out);
    int ok = (maxulp <= 8) || (maxabs < 1e-4);
    printf("  %s\n", ok ? "*** ATOM VALIDATED: GPU recompute matches Python reference (bounded) ***"
                        : "!!! DIVERGENCE — investigate");
    return ok ? 0 : 1;
}
