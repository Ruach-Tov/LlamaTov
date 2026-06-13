// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
// soa_repack_verify.c — isolate the repack + kernel-read offset arithmetic, check it
// reconstructs the original AoS values. Finds the offset mismatch WITHOUT the GPU.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

int main(){
    // one row, bpr blocks, AoS = [d(2)|qs(32)] x bpr
    int bpr = 4;                     // small, K=128
    int row_bytes = bpr*34;
    uint8_t aos[row_bytes];
    // fill deterministic: block b: scale bytes = (b,b+100), quants = b*10 + i
    for(int b=0;b<bpr;b++){
        aos[b*34+0]=(uint8_t)b; aos[b*34+1]=(uint8_t)(b+100);
        for(int i=0;i<32;i++) aos[b*34+2+i]=(uint8_t)(b*10+i);
    }

    // --- REPACK (mavchin's set_tensor logic) ---
    uint8_t soa[row_bytes];
    for(int b=0;b<bpr;b++){
        memcpy(soa + b*32,            aos + b*34 + 2, 32);  // quants
        memcpy(soa + bpr*32 + b*2,    aos + b*34,     2);   // scale
    }

    // --- KERNEL READ (what the SoA vec_dot SHOULD do) ---
    // row_base = soa. quants_ptr(b) = row_base + b*32. scales_ptr(b) = row_base + bpr*32 + b*2.
    printf("=== SoA repack + kernel-read verification (bpr=%d) ===\n", bpr);
    int ok=1;
    for(int b=0;b<bpr;b++){
        const uint8_t* q = soa + b*32;
        const uint8_t* sc = soa + bpr*32 + b*2;
        // compare to original AoS values
        int qok=1; for(int i=0;i<32;i++) if(q[i]!=aos[b*34+2+i]) qok=0;
        int scok = (sc[0]==aos[b*34+0] && sc[1]==aos[b*34+1]);
        printf("  block %d: quants %s, scale %s (read q[0]=%d expect %d; sc=(%d,%d) expect (%d,%d))\n",
               b, qok?"OK":"MISMATCH", scok?"OK":"MISMATCH",
               q[0], aos[b*34+2], sc[0], sc[1], aos[b*34+0], aos[b*34+1]);
        if(!qok||!scok) ok=0;
    }
    printf("\n  REPACK+READ %s\n", ok?"CORRECT (offset arithmetic is sound)":"BROKEN (offset mismatch HERE)");
    printf("\n  => If this is CORRECT, the bug is NOT the single-row offset math.\n");
    printf("     Then suspect: (1) stride_row_x in the kernel still assumes AoS 34B stride,\n");
    printf("         so row0*stride_row_x jumps to the wrong SoA row base.\n");
    printf("     (2) the kernel reads vbq+kbx as block_q8_0* (34B struct) instead of\n");
    printf("         SoA quants_ptr = row_base + kbx*32 / scales = row_base + bpr*32 + kbx*2.\n");
    printf("     (3) the activation (Q8_1 y) path unchanged but mis-indexed relative to SoA x.\n");
    return 0;
}
