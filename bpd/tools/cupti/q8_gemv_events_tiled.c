// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* q8_gemv_events.c — measure sectors_per_load for OUR k_q8_0_gemv via CUPTI event counters.
 * Mirrors Mavchin's ggml_cupti_events.c event-group sequence, but launches OUR kernel (cubin)
 * instead of ggml's matmul. One event group per event (Pascal sm_61 needs separate groups).
 *
 *   sectors_per_load = (fb_subp0_read_sectors + fb_subp1_read_sectors) / sum(gld_inst_*bit)
 *   ~1   -> loads coalesced (ggml-efficient; layout rewrite wasted)
 *   ~2+  -> 34B Q8_0 block straddles cache lines -> SoA layout helps
 *
 * Prolog: run_q8_gemv_events(+Cubin, +M, +K, -SectorsPerLoad).
 */
#include <cuda.h>
#include <cupti.h>
#include <SWI-Prolog.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* ev_names[] = {
  "fb_subp0_read_sectors", "fb_subp1_read_sectors",
  "gld_inst_8bit", "gld_inst_16bit", "gld_inst_32bit",
  "gld_inst_64bit", "gld_inst_128bit"
};
#define N_EV 7

static foreign_t pl_run_q8_gemv_events(term_t t_cubin, term_t t_m, term_t t_k, term_t t_spl) {
  char* cubin; int M, K;
  if (!PL_get_atom_chars(t_cubin, &cubin)) PL_fail;
  if (!PL_get_integer(t_m, &M)) PL_fail;
  if (!PL_get_integer(t_k, &K)) PL_fail;
  int nb = K / 32;

  cuInit(0);
  CUdevice dev; cuDeviceGet(&dev, 0);
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  if (!ctx) cuCtxCreate(&ctx, 0, dev);

  CUmodule mod; CUfunction fn;
  if (cuModuleLoad(&mod, cubin) != CUDA_SUCCESS) { printf("cuModuleLoad failed\n"); PL_fail; }
  if (cuModuleGetFunction(&fn, mod, "k_q8_0_gemv") != CUDA_SUCCESS) { printf("no k_q8_0_gemv\n"); PL_fail; }

  CUdeviceptr Wq, Wd, Xq, Xd, Y;
  cuMemAlloc(&Wq, (size_t)M*K); cuMemAlloc(&Wd, (size_t)M*nb*2);
  cuMemAlloc(&Xq, (size_t)K);   cuMemAlloc(&Xd, (size_t)nb*2);
  cuMemAlloc(&Y,  (size_t)M*4);
  void* args[] = { &Wq, &Wd, &Xq, &Xd, &Y, &M, &K };
  unsigned BM = 16; unsigned blk = BM*32; unsigned grid = (M + BM - 1) / BM; unsigned shmem = K + nb*2;  /* tiled v4 geometry */

  /* warmup */
  for (int i=0;i<5;i++) cuLaunchKernel(fn, grid,1,1, blk,1,1, shmem, 0, args, 0);
  cuCtxSynchronize();

  uint64_t vals[N_EV];
  for (int e=0;e<N_EV;e++) {
    vals[e]=0;
    CUpti_EventID id;
    if (cuptiEventGetIdFromName(dev, ev_names[e], &id) != CUPTI_SUCCESS) { vals[e]=(uint64_t)-1; continue; }
    CUpti_EventGroup grp;
    if (cuptiEventGroupCreate(ctx, &grp, 0) != CUPTI_SUCCESS) { vals[e]=(uint64_t)-2; continue; }
    if (cuptiEventGroupAddEvent(grp, id) != CUPTI_SUCCESS) { cuptiEventGroupDestroy(grp); vals[e]=(uint64_t)-3; continue; }
    uint32_t all=1;
    cuptiEventGroupSetAttribute(grp, CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES, sizeof(all), &all);
    cuptiEventGroupEnable(grp);
    cuLaunchKernel(fn, grid,1,1, blk,1,1, shmem, 0, args, 0);
    cuCtxSynchronize();
    size_t valsSz=sizeof(uint64_t)*64, idsSz=sizeof(CUpti_EventID)*64, numRead=0;
    uint64_t rv[64]; CUpti_EventID rid[64];
    if (cuptiEventGroupReadAllEvents(grp, CUPTI_EVENT_READ_FLAG_NONE, &valsSz, rv, &idsSz, rid, &numRead)==CUPTI_SUCCESS && numRead>0) {
      uint64_t s=0; for (size_t i=0;i<numRead;i++) s+=rv[i];   /* sum domain instances */
      vals[e]=s;
    }
    cuptiEventGroupDisable(grp);
    cuptiEventGroupDestroy(grp);
  }

  uint64_t fb = vals[0]+vals[1];
  uint64_t gld = vals[2]+vals[3]+vals[4]+vals[5]+vals[6];
  double spl = gld>0 ? (double)fb/(double)gld : -1.0;

  printf("=== k_q8_0_gemv event counters  M=%d K=%d ===\n", M, K);
  for (int e=0;e<N_EV;e++) printf("  %-26s %ld\n", ev_names[e], (long)vals[e]);
  printf("  --- fb_read_sectors=%lu  gld_inst=%lu ---\n", (unsigned long)fb, (unsigned long)gld);
  printf("  SECTORS PER LOAD = %.4f\n", spl);

  cuMemFree(Wq);cuMemFree(Wd);cuMemFree(Xq);cuMemFree(Xd);cuMemFree(Y);
  cuModuleUnload(mod);

  term_t spl_t = PL_new_term_ref();
  PL_put_float(spl_t, spl);
  return PL_unify(t_spl, spl_t);
}

install_t install_q8_gemv_events(void) {
  PL_register_foreign("run_q8_gemv_events", 4, pl_run_q8_gemv_events, 0);
}
