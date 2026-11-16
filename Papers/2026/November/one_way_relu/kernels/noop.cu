extern "C" __global__ void noop_copy(const float* in, float* out, int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i];}
