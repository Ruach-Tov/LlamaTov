// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* mlir_gpu_launch.cu — launch the MLIR-generated relu PTX on the P4 via the
 * CUDA driver API. Proves MLIR -> NVVM -> PTX -> P4 end-to-end.
 *
 * The MLIR memref<?xf32> ABI for a kernel arg expands to 5 scalars:
 *   allocated_ptr (u64), aligned_ptr (u64), offset (i64), size (i64), stride (i64)
 * Our kernel @relu(%x: memref, %c: memref) -> 10 params (relu_param_0..9):
 *   x: alloc,align,offset,size,stride  (params 0-4)
 *   c: alloc,align,offset,size,stride  (params 5-9)
 * From the PTX: it reads param_3 (x.size) for bounds, param_1 (x.aligned),
 * param_6 (c.aligned). So we must pass aligned_ptr and size correctly.
 */
#include <cstdio>
#include <cstdint>
#include <cuda.h>

#define CK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s: %s\n",#x,s);return 1;} } while(0)

int main() {
    const int N = 1024;
    // read input.bin (the same reference input as the other backends)
    float xh[N], ch[N];
    FILE* fi = fopen("/tmp/gpu-work/referee/input.bin","rb");
    fread(xh, 4, N, fi); fclose(fi);

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev, 0));
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));
    CUmodule mod; CK(cuModuleLoadData(&mod, /*PTX*/ ({
        FILE* fp=fopen("/tmp/gpu-work/mlir_backend/relu_mlir.ptx","rb");
        fseek(fp,0,SEEK_END); long sz=ftell(fp); fseek(fp,0,SEEK_SET);
        static char buf[1<<16]; fread(buf,1,sz,fp); buf[sz]=0; fclose(fp); buf;
    })));
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, "relu"));

    CUdeviceptr xd, cd;
    CK(cuMemAlloc(&xd, N*4)); CK(cuMemAlloc(&cd, N*4));
    CK(cuMemcpyHtoD(xd, xh, N*4));

    // memref ABI args: x{alloc,align,offset,size,stride}, c{alloc,align,offset,size,stride}
    int64_t zero=0, n=N, one=1;
    void* args[] = {
        &xd,&xd,&zero,&n,&one,   // x memref
        &cd,&cd,&zero,&n,&one    // c memref
    };
    int blk=256, grd=(N+blk-1)/blk;
    CK(cuLaunchKernel(fn, grd,1,1, blk,1,1, 0,0, args, 0));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(ch, cd, N*4));

    // write output for the referee
    FILE* fo=fopen("/tmp/gpu-work/referee/relu_mlir_gpu.bin","wb");
    fwrite(ch,4,N,fo); fclose(fo);
    printf("MLIR-GPU relu launched on P4. sample: in[4]=%g out[4]=%g  in[3]=%g out[3]=%g\n",
           xh[4], ch[4], xh[3], ch[3]);
    return 0;
}
