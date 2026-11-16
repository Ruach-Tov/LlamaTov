// CANONICAL one-way-leaky backward — CUDA, emitted from op_expr (var2) by expr_ir.
// grad_in = grad_out * ((x>0)?1 : (x<=0 && g<0)?alpha : 0); x=0 DEAD; NaN->dead.
extern "C" __global__ void oneway_backward(const float* x, const float* g, float* dst, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    float xv = x[i], gv = g[i];
    float mask = (xv > 0.0f) ? 1.0f : ((gv < 0.0f) ? 0.01f : 0.0f);
    dst[i] = gv * mask;
  }
}
