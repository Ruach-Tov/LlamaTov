// SPDX-License-Identifier: LicenseRef-RTAAL-1.1
// rc_rope_harness.cu — STEP 2 validation: run the emitted k_rope on a real K and compare to the Python
// reference K_ref (RoPE at position POS). CRITICAL: k_rope uses pos=blockIdx.x. For a single decode token
// at position POS, the kernel must rotate at POS, not 0. We test BOTH ways to expose the behavior:
//   (A) naive single-token launch <<<1, ...>>>  -> blockIdx.x=0 -> rotates at pos 0 (WRONG for decode)
//   (B) position-correct: we pad/offset so the token's grid pos = POS (or pass POS explicitly).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

// the emitted rope kernel (uses pos = blockIdx.x)
__global__ void k_rope(float * __restrict__ q, int seq_len, int n_head, int head_dim, float theta_base) {
    int pos = blockIdx.x;
    int h = blockIdx.y;
    if (pos >= seq_len) return;
    int half = head_dim / 2;
    float * q_head = q + (((pos * n_head) + h) * head_dim);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.0f / powf(theta_base, ((float)(2 * i)) / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle), sin_a = sinf(angle);
        float q0 = q_head[i], q1 = q_head[i + half];
        q_head[i] = (q0 * cos_a) - (q1 * sin_a);
        q_head[i + half] = (q0 * sin_a) + (q1 * cos_a);
    }
}

static float* slurp(const char* p, size_t n){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(2);}float*b=(float*)malloc(n);fread(b,1,n,f);fclose(f);return b;}

static int compare(const char* tag, float* got, float* ref, int n){
    int exact=0; double maxabs=0;
    for(int i=0;i<n;i++){double d=fabs((double)got[i]-(double)ref[i]); if(d>maxabs)maxabs=d; if(d==0)exact++;}
    int ok = maxabs < 1e-4;
    printf("  [%s] got[0..3]=%.5f %.5f %.5f %.5f  exact=%d/%d  max_abs=%.3e  %s\n",
           tag, got[0],got[1],got[2],got[3], exact, n, maxabs, ok?"MATCH":"DIFFER");
    return ok;
}

int main(int argc, char** argv){
    int n_head_k=atoi(argv[1]), head_dim=atoi(argv[2]); float theta=atof(argv[3]); int POS=atoi(argv[4]);
    int n = n_head_k*head_dim;
    float* Kin = slurp("/tmp/fx_rope_kin.bin", (size_t)n*4);
    float* Kref= slurp("/tmp/fx_rope_kref.bin",(size_t)n*4);
    printf("=== k_rope STEP-2 validation: n_head_k=%d head_dim=%d theta=%g POS=%d ===\n", n_head_k, head_dim, theta, POS);

    // (A) naive single-token launch: grid.x = 1 -> pos=0 (the prefill-only assumption)
    {
        float* dk; cudaMalloc(&dk,(size_t)n*4); cudaMemcpy(dk,Kin,(size_t)n*4,cudaMemcpyHostToDevice);
        dim3 grid(1, n_head_k); k_rope<<<grid, 64>>>(dk, 1, n_head_k, head_dim, theta);
        cudaDeviceSynchronize();
        float* out=(float*)malloc((size_t)n*4); cudaMemcpy(out,dk,(size_t)n*4,cudaMemcpyDeviceToHost);
        compare("A: naive <<<1,...>>> (pos=blockIdx.x=0)", out, Kref, n);
        cudaFree(dk); free(out);
    }
    // (B) position-correct: launch grid.x = POS+1 so the token sits at grid row POS, and offset the buffer
    //     so q_head computes the same address. We put the token at pos index POS in a (POS+1)-row buffer.
    {
        int rows = POS+1;
        float* dk; cudaMalloc(&dk,(size_t)rows*n*4); cudaMemset(dk,0,(size_t)rows*n*4);
        // place K at row POS
        cudaMemcpy(dk + (size_t)POS*n, Kin, (size_t)n*4, cudaMemcpyHostToDevice);
        dim3 grid(rows, n_head_k); k_rope<<<grid, 64>>>(dk, rows, n_head_k, head_dim, theta);
        cudaDeviceSynchronize();
        float* out=(float*)malloc((size_t)n*4); cudaMemcpy(out, dk + (size_t)POS*n, (size_t)n*4, cudaMemcpyDeviceToHost);
        compare("B: pos-correct (token at grid row POS)", out, Kref, n);
        cudaFree(dk); free(out);
    }
    return 0;
}
