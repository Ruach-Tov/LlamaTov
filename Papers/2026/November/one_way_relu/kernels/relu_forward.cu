extern "C" __global__ void relu_forward(const float* x, float* dst, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; dst[i] = (v>=0.0f) ? v : 0.0f; }}
