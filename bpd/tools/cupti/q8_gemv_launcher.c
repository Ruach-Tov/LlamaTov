// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* q8_gemv_launcher.c — a Prolog foreign predicate that launches OUR k_q8_0_gemv so the
 * cupti-from-prolog stall instrument has work to sample. Compiled INTO the same .so as
 * cupti_bridge.c + bpd_cupti_profile.c, so init -> run_q8_gemv -> flush -> stall_report all
 * happen in one swipl process (Mavchin's working pattern, driven from Prolog as intended).
 *
 * This is a SEPARATE file (does not modify Mavchin's cupti_bridge.c) — it only adds a
 * kernel-launch predicate for our GEMV.
 *
 *   ?- cupti_init, run_q8_gemv("path.cubin", 896, 4864, 200), cupti_flush,
 *      cupti_stall_report(Stalls).
 */
#include <cuda.h>
#include <SWI-Prolog.h>
#include <stdio.h>
#include <stdlib.h>

static foreign_t pl_run_q8_gemv(term_t t_cubin, term_t t_m, term_t t_k, term_t t_iters) {
  char* cubin; int M, K, iters;
  if (!PL_get_atom_chars(t_cubin, &cubin)) PL_fail;
  if (!PL_get_integer(t_m, &M)) PL_fail;
  if (!PL_get_integer(t_k, &K)) PL_fail;
  if (!PL_get_integer(t_iters, &iters)) PL_fail;
  int nb = K / 32;

  /* CUDA context is already current (cupti_init created/uses it). Ensure init. */
  cuInit(0);
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  if (!ctx) {
    CUdevice dev; cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev);
  }
  CUmodule mod; CUfunction fn;
  if (cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) { printf("cuModuleLoad failed: %s\n", cubin); PL_fail; }
  if (cuModuleGetFunction(&fn, mod, "k_q8_0_gemv") != CUDA_SUCCESS) { printf("no k_q8_0_gemv\n"); PL_fail; }

  CUdeviceptr Wq, Wd, Xq, Xd, Y;
  cuMemAlloc(&Wq, (size_t)M*K);
  cuMemAlloc(&Wd, (size_t)M*nb*2);
  cuMemAlloc(&Xq, (size_t)K);
  cuMemAlloc(&Xd, (size_t)nb*2);
  cuMemAlloc(&Y,  (size_t)M*4);
  void* args[] = { &Wq, &Wd, &Xq, &Xd, &Y, &M, &K };
  unsigned blk = 64, grid = (M + blk - 1) / blk;

  for (int i = 0; i < iters; i++)
    cuLaunchKernel(fn, grid,1,1, blk,1,1, 0, 0, args, 0);
  cuCtxSynchronize();

  cuMemFree(Wq); cuMemFree(Wd); cuMemFree(Xq); cuMemFree(Xd); cuMemFree(Y);
  cuModuleUnload(mod);
  PL_succeed;
}

/* TILED variant: same kernel name (k_q8_0_gemv), but the tiled launch geometry —
 * grid=(M+BM-1)/BM, block=BM*32, dynamic shared = (K + nb*2) bytes (the staged activation).
 * Lets the stall sampler profile the TILED GEMV (the serial launcher's grid/blk/shmem=0 is
 * wrong for it). run_q8_gemv_tiled("path.cubin", M, K, BM, Iters). */
static foreign_t pl_run_q8_gemv_tiled(term_t t_cubin, term_t t_m, term_t t_k,
                                      term_t t_bm, term_t t_iters) {
  char* cubin; int M, K, BM, iters;
  if (!PL_get_atom_chars(t_cubin, &cubin)) PL_fail;
  if (!PL_get_integer(t_m, &M)) PL_fail;
  if (!PL_get_integer(t_k, &K)) PL_fail;
  if (!PL_get_integer(t_bm, &BM)) PL_fail;
  if (!PL_get_integer(t_iters, &iters)) PL_fail;
  int nb = K / 32;
  cuInit(0);
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  if (!ctx) { CUdevice dev; cuDeviceGet(&dev, 0); cuCtxCreate(&ctx, 0, dev); }
  CUmodule mod; CUfunction fn;
  if (cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) { printf("cuModuleLoad failed: %s\n", cubin); PL_fail; }
  if (cuModuleGetFunction(&fn, mod, "k_q8_0_gemv") != CUDA_SUCCESS) { printf("no k_q8_0_gemv\n"); PL_fail; }
  CUdeviceptr Wq, Wd, Xq, Xd, Y;
  cuMemAlloc(&Wq, (size_t)M*K);
  cuMemAlloc(&Wd, (size_t)M*nb*2);
  cuMemAlloc(&Xq, (size_t)K);
  cuMemAlloc(&Xd, (size_t)nb*2);
  cuMemAlloc(&Y,  (size_t)M*4);
  void* args[] = { &Wq, &Wd, &Xq, &Xd, &Y, &M, &K };
  unsigned blk = (unsigned)BM * 32;
  unsigned grid = (M + BM - 1) / BM;
  unsigned shmem = (unsigned)(K + nb*2);   /* staged activation: K int8 + nb fp16 scales */
  for (int i = 0; i < iters; i++)
    cuLaunchKernel(fn, grid,1,1, blk,1,1, shmem, 0, args, 0);
  cuCtxSynchronize();
  cuMemFree(Wq); cuMemFree(Wd); cuMemFree(Xq); cuMemFree(Xd); cuMemFree(Y);
  cuModuleUnload(mod);
  PL_succeed;
}

install_t install_q8_gemv_launcher(void) {
  PL_register_foreign("run_q8_gemv", 4, pl_run_q8_gemv, 0);
  PL_register_foreign("run_q8_gemv_tiled", 5, pl_run_q8_gemv_tiled, 0);
}
