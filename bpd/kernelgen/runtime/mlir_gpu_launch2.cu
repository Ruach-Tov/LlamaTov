// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* mlir_gpu_launch2.cu — generic MLIR-GPU launcher with libdevice linking.
 * Usage: mlir_gpu_launch2 <op> <ptx_path> <out_bin>
 * Links libdevice.10.bc via cuLink so __nv_* (tanhf/erff/expf) resolve.
 * Reads /tmp/gpu-work/referee/input.bin, launches <op> kernel,
 * writes <out_bin>. memref ABI: {ptr,ptr,0,N,1} x2.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda.h>

#define CK(x) do{CUresult r=(x);if(r!=CUDA_SUCCESS){const char*s;cuGetErrorString(r,&s);printf("ERR %s: %s\n",#x,s);return 1;}}while(0)

static char* slurp(const char* p, size_t* n){
    FILE* f=fopen(p,"rb"); if(!f){printf("open fail %s\n",p);exit(1);} 
    fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET);
    char* b=(char*)malloc(s+1); fread(b,1,s,f); b[s]=0; fclose(f); if(n)*n=s; return b;
}

int main(int argc, char** argv){
    const char* op=argv[1]; const char* ptxp=argv[2]; const char* outp=argv[3];
    const char* libdevice="/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8/nvvm/libdevice/libdevice.10.bc";
    const int N=1024;
    float xh[N], ch[N];
    size_t isz; char* ib=slurp("/tmp/gpu-work/referee/input.bin",&isz);
    memcpy(xh, ib, N*4);

    CK(cuInit(0));
    CUdevice dev; CK(cuDeviceGet(&dev,0));
    CUcontext ctx; CK(cuCtxCreate(&ctx,0,dev));

    size_t ptxn; char* ptx=slurp(ptxp,&ptxn);
    CUmodule mod;
    // Link PTX + libdevice.bc via cuLink (resolves __nv_*).
    CUlinkState ls; 
    CUjit_option opts[]={CU_JIT_TARGET}; void* ovals[]={(void*)(uintptr_t)CU_TARGET_COMPUTE_61};
    CK(cuLinkCreate(1,opts,ovals,&ls));
    // add libdevice bitcode
    size_t ldn; char* ld=slurp(libdevice,&ldn);
    CK(cuLinkAddData(ls,CU_JIT_INPUT_LIBRARY,ld,ldn,"libdevice",0,0,0));
    // add our PTX
    CK(cuLinkAddData(ls,CU_JIT_INPUT_PTX,ptx,ptxn+1,"kernel.ptx",0,0,0));
    void* cubin; size_t cubinsz;
    CK(cuLinkComplete(ls,&cubin,&cubinsz));
    CK(cuModuleLoadData(&mod,cubin));
    CUfunction fn; CK(cuModuleGetFunction(&fn,mod,op));

    CUdeviceptr xd,cd; CK(cuMemAlloc(&xd,N*4)); CK(cuMemAlloc(&cd,N*4));
    CK(cuMemcpyHtoD(xd,xh,N*4));
    int64_t zero=0,n=N,one=1;
    void* args[]={&xd,&xd,&zero,&n,&one, &cd,&cd,&zero,&n,&one};
    int blk=256, grd=(N+blk-1)/blk;
    CK(cuLaunchKernel(fn,grd,1,1, blk,1,1, 0,0, args,0));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(ch,cd,N*4));
    FILE* fo=fopen(outp,"wb"); fwrite(ch,4,N,fo); fclose(fo);
    printf("MLIR-GPU %s launched on P4 (libdevice linked). in[4]=%g out[4]=%g\n", op, xh[4], ch[4]);
    cuLinkDestroy(ls);
    return 0;
}
