// PROTOTYPE one-way-leaky backward — CUDA/C (bocher, 2026-11-16).
// Same semantics as oneway_backward_cpu.ll (relu doc §7.3):
//   grad_in = g * ( (x>0) ? 1 : (g<0) ? alpha : 0 )
// Convention header (§6): x==0 (incl -0.0) = DEAD side; NaN x -> mask 0
// (ogt false). alpha = 0.01f baked (oneway_10 arm).
// Elementwise map, grid-stride loop, coalesced. sm_61 target.
extern "C" __global__
void oneway_backward(const float* __restrict__ x,
                     const float* __restrict__ g,
                     float* __restrict__ dst, long long n)
{
    const float alpha = 0.01f;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n;
         i += (long long)gridDim.x * blockDim.x) {
        float xv = x[i];
        float gv = g[i];
        float deadf = (gv < 0.0f) ? alpha : 0.0f;
        float mask  = (xv > 0.0f) ? 1.0f : deadf;
        dst[i] = gv * mask;
    }
}
