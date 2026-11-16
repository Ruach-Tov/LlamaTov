extern "C" __global__ void strict_backward(const float* x, const float* g, float* dst, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ dst[i] = (x[i]>0.0f) ? g[i] : 0.0f; }}
