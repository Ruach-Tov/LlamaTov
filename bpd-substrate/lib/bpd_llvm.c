// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_llvm_v2.c — Prolog→LLVM IR emitter for vec_dot.
 *
 * Sweepable parameters from Prolog:
 *   ARR: number of vector accumulators (4, 8)
 *   FMA: 0=fmul+fadd, 1=llvm.fma
 *
 * Fixed: uses <4 x float> vector type (matches SSE/NEON).
 */

#include <SWI-Prolog.h>
#include <llvm-c/Core.h>
#include <llvm-c/Analysis.h>
#include <llvm-c/BitWriter.h>
#include <stdio.h>
#include <string.h>

static foreign_t pl_llvm_create_module(term_t name_t, term_t mod_t) {
    char *name;
    if (!PL_get_atom_chars(name_t, &name)) return FALSE;
    return PL_unify_pointer(mod_t, LLVMModuleCreateWithName(name));
}

static foreign_t pl_llvm_dispose_module(term_t mod_t) {
    void *p; if (!PL_get_pointer(mod_t, &p)) return FALSE;
    LLVMDisposeModule((LLVMModuleRef)p); return TRUE;
}

static foreign_t pl_llvm_dump_ir(term_t mod_t) {
    void *p; if (!PL_get_pointer(mod_t, &p)) return FALSE;
    char *ir = LLVMPrintModuleToString((LLVMModuleRef)p);
    printf("%s\n", ir);
    LLVMDisposeMessage(ir);
    return TRUE;
}

static foreign_t pl_llvm_write_bitcode(term_t mod_t, term_t path_t) {
    void *p; char *path;
    if (!PL_get_pointer(mod_t, &p)) return FALSE;
    if (!PL_get_atom_chars(path_t, &path)) return FALSE;
    return LLVMWriteBitcodeToFile((LLVMModuleRef)p, path) == 0;
}

/* emit vec_dot: void fn(i32 n, float* s, float* x, float* y)
 * Uses <4 x float> vectors, ARR accumulators, optional FMA. */
static foreign_t pl_llvm_emit_vec_dot(term_t mod_t, term_t arr_t,
                                       term_t fma_t, term_t name_t) {
    void *mod_ptr; int arr, use_fma; char *fn_name;
    if (!PL_get_pointer(mod_t, &mod_ptr)) return FALSE;
    if (!PL_get_integer(arr_t, &arr)) return FALSE;
    if (!PL_get_integer(fma_t, &use_fma)) return FALSE;
    if (!PL_get_atom_chars(name_t, &fn_name)) return FALSE;

    LLVMModuleRef mod = (LLVMModuleRef)mod_ptr;
    LLVMContextRef ctx = LLVMGetModuleContext(mod);

    int step = 4 * arr;  /* 4 floats per vector * arr accumulators */

    LLVMTypeRef f32    = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef v4f32  = LLVMVectorType(f32, 4);  /* <4 x float> */
    LLVMTypeRef i32    = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64    = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef fptr   = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef voidty = LLVMVoidTypeInContext(ctx);

    /* void fn(i32 %n, ptr %s, ptr %x, ptr %y) */
    LLVMTypeRef params[] = {i32, fptr, fptr, fptr};
    LLVMTypeRef fn_type = LLVMFunctionType(voidty, params, 4, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, fn_name, fn_type);
    LLVMSetValueName2(LLVMGetParam(fn, 0), "n", 1);
    LLVMSetValueName2(LLVMGetParam(fn, 1), "s", 1);
    LLVMSetValueName2(LLVMGetParam(fn, 2), "x", 1);
    LLVMSetValueName2(LLVMGetParam(fn, 3), "y", 1);

    LLVMBuilderRef B = LLVMCreateBuilderInContext(ctx);

    LLVMBasicBlockRef bb_entry  = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef bb_loop   = LLVMAppendBasicBlockInContext(ctx, fn, "loop");
    LLVMBasicBlockRef bb_reduce = LLVMAppendBasicBlockInContext(ctx, fn, "reduce");
    LLVMBasicBlockRef bb_done   = LLVMAppendBasicBlockInContext(ctx, fn, "done");

    LLVMValueRef n_val = LLVMGetParam(fn, 0);
    LLVMValueRef s_ptr = LLVMGetParam(fn, 1);
    LLVMValueRef x_ptr = LLVMGetParam(fn, 2);
    LLVMValueRef y_ptr = LLVMGetParam(fn, 3);

    LLVMValueRef zero_v  = LLVMConstNull(v4f32);
    LLVMValueRef zero_f  = LLVMConstReal(f32, 0.0);
    LLVMValueRef zero_i  = LLVMConstInt(i32, 0, 0);
    LLVMValueRef step_v  = LLVMConstInt(i32, step, 0);

    /* ---- entry ---- */
    LLVMPositionBuilderAtEnd(B, bb_entry);
    LLVMValueRef mask = LLVMConstInt(i32, ~(step - 1), 0);
    LLVMValueRef np   = LLVMBuildAnd(B, n_val, mask, "np");
    LLVMValueRef cmp  = LLVMBuildICmp(B, LLVMIntSGT, np, zero_i, "has_main");
    LLVMBuildCondBr(B, cmp, bb_loop, bb_done);

    /* ---- loop ---- */
    LLVMPositionBuilderAtEnd(B, bb_loop);
    LLVMValueRef i_phi = LLVMBuildPhi(B, i32, "i");

    /* ARR vector accumulators */
    LLVMValueRef acc[16], new_acc[16];  /* max arr=16 */
    char nm[32];
    for (int j = 0; j < arr; j++) {
        snprintf(nm, sizeof(nm), "sum%d", j);
        acc[j] = LLVMBuildPhi(B, v4f32, nm);
    }

    /* Loop body: load <4 x float> from x and y, accumulate */
    for (int j = 0; j < arr; j++) {
        int offset = j * 4;
        LLVMValueRef off_val = LLVMConstInt(i32, offset, 0);
        LLVMValueRef idx = LLVMBuildAdd(B, i_phi, off_val, "idx");
        LLVMValueRef idx64 = LLVMBuildSExt(B, idx, i64, "idx64");

        /* GEP to float*, then load as <4 x float> */
        LLVMValueRef xg = LLVMBuildGEP2(B, f32, x_ptr, &idx64, 1, "xg");
        LLVMValueRef yg = LLVMBuildGEP2(B, f32, y_ptr, &idx64, 1, "yg");

        LLVMValueRef xv = LLVMBuildLoad2(B, v4f32, xg, "xv");
        LLVMValueRef yv = LLVMBuildLoad2(B, v4f32, yg, "yv");
        LLVMSetAlignment(xv, 4);
        LLVMSetAlignment(yv, 4);

        if (use_fma) {
            /* llvm.fma.v4f32 */
            LLVMTypeRef ovl[] = {v4f32};
            unsigned fma_id = LLVMLookupIntrinsicID("llvm.fma", 8);
            LLVMValueRef fma_fn = LLVMGetIntrinsicDeclaration(mod, fma_id, ovl, 1);
            LLVMTypeRef fma_ty = LLVMIntrinsicGetType(ctx, fma_id, ovl, 1);
            LLVMValueRef args[] = {xv, yv, acc[j]};
            snprintf(nm, sizeof(nm), "fma%d", j);
            new_acc[j] = LLVMBuildCall2(B, fma_ty, fma_fn, args, 3, nm);
        } else {
            /* fmul + fadd */
            snprintf(nm, sizeof(nm), "mul%d", j);
            LLVMValueRef mul = LLVMBuildFMul(B, xv, yv, nm);
            snprintf(nm, sizeof(nm), "add%d", j);
            new_acc[j] = LLVMBuildFAdd(B, acc[j], mul, nm);
        }
    }

    /* Loop increment */
    LLVMValueRef i_next = LLVMBuildAdd(B, i_phi, step_v, "i_next");
    LLVMValueRef again  = LLVMBuildICmp(B, LLVMIntSLT, i_next, np, "again");
    LLVMBuildCondBr(B, again, bb_loop, bb_reduce);

    /* Wire loop PHIs */
    LLVMValueRef i_in[]   = {zero_i, i_next};
    LLVMBasicBlockRef i_bb[] = {bb_entry, bb_loop};
    LLVMAddIncoming(i_phi, i_in, i_bb, 2);
    for (int j = 0; j < arr; j++) {
        LLVMValueRef v[] = {zero_v, new_acc[j]};
        LLVMAddIncoming(acc[j], v, i_bb, 2);
    }

    /* ---- reduce ---- */
    LLVMPositionBuilderAtEnd(B, bb_reduce);

    /* Binary tree: acc[0]+=acc[arr/2], ... */
    LLVMValueRef red[16];
    for (int j = 0; j < arr; j++) red[j] = new_acc[j];

    int half = arr >> 1;
    while (half > 0) {
        for (int j = 0; j < half; j++) {
            snprintf(nm, sizeof(nm), "r%d", j);
            red[j] = LLVMBuildFAdd(B, red[j], red[j + half], nm);
        }
        half >>= 1;
    }

    /* Horizontal sum of <4 x float> → float */
    /* Extract lanes 0,1,2,3 and add sequentially */
    LLVMValueRef e0 = LLVMBuildExtractElement(B, red[0], LLVMConstInt(i32, 0, 0), "e0");
    LLVMValueRef e1 = LLVMBuildExtractElement(B, red[0], LLVMConstInt(i32, 1, 0), "e1");
    LLVMValueRef e2 = LLVMBuildExtractElement(B, red[0], LLVMConstInt(i32, 2, 0), "e2");
    LLVMValueRef e3 = LLVMBuildExtractElement(B, red[0], LLVMConstInt(i32, 3, 0), "e3");
    LLVMValueRef h1 = LLVMBuildFAdd(B, e0, e1, "h1");
    LLVMValueRef h2 = LLVMBuildFAdd(B, h1, e2, "h2");
    LLVMValueRef h3 = LLVMBuildFAdd(B, h2, e3, "h3");

    LLVMBuildStore(B, h3, s_ptr);
    LLVMBuildRetVoid(B);

    /* ---- done (skip main loop) ---- */
    LLVMPositionBuilderAtEnd(B, bb_done);
    LLVMBuildStore(B, zero_f, s_ptr);
    LLVMBuildRetVoid(B);

    LLVMDisposeBuilder(B);
    return TRUE;
}

install_t install_bpd_llvm(void) {
    PL_register_foreign("llvm_create_module",  2, pl_llvm_create_module, 0);
    PL_register_foreign("llvm_dispose_module", 1, pl_llvm_dispose_module, 0);
    PL_register_foreign("llvm_dump_ir",        1, pl_llvm_dump_ir, 0);
    PL_register_foreign("llvm_write_bitcode",  2, pl_llvm_write_bitcode, 0);
    PL_register_foreign("llvm_emit_vec_dot",   4, pl_llvm_emit_vec_dot, 0);
}
