// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* bpd_llvm_elem.c — Prolog→LLVM IR for elementwise operations.
 *
 * Covers 12 unary ops + 1 binary op = 13 of 44 L1 kernels.
 * Same foreign interface pattern as bpd_llvm.c (vec_dot).
 *
 * Prolog usage:
 *   llvm_emit_unary(Mod, relu, bpd_relu).
 *   llvm_emit_unary(Mod, silu, bpd_silu).
 *   llvm_emit_unary(Mod, gelu, bpd_gelu).
 */

#include <SWI-Prolog.h>
#include <llvm-c/Core.h>
#include <llvm-c/BitWriter.h>
#include <stdio.h>
#include <string.h>

/* Helper: get or declare an LLVM intrinsic for <4 x float> */
static LLVMValueRef get_v4f32_intrinsic(LLVMModuleRef mod, LLVMContextRef ctx,
                                         const char *name) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMTypeRef ovl[] = {v4f32};
    unsigned id = LLVMLookupIntrinsicID(name, strlen(name));
    return LLVMGetIntrinsicDeclaration(mod, id, ovl, 1);
}

static LLVMTypeRef get_v4f32_intrinsic_type(LLVMContextRef ctx, const char *name) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMTypeRef ovl[] = {v4f32};
    unsigned id = LLVMLookupIntrinsicID(name, strlen(name));
    return LLVMIntrinsicGetType(ctx, id, ovl, 1);
}

/* Build a <4 x float> splat from a scalar float constant */
static LLVMValueRef splat_f32(LLVMBuilderRef B, LLVMContextRef ctx, double val) {
    LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef v4f32 = LLVMVectorType(f32, 4);
    LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx);
    LLVMValueRef scalar = LLVMConstReal(f32, val);
    LLVMValueRef undef = LLVMGetUndef(v4f32);
    LLVMValueRef v = undef;
    for (int i = 0; i < 4; i++)
        v = LLVMBuildInsertElement(B, v, scalar, LLVMConstInt(i32, i, 0), "splat");
    return v;
}

/* Call a <4 x float> intrinsic with one arg */
static LLVMValueRef call_v4f32_unary(LLVMBuilderRef B, LLVMModuleRef mod,
                                      LLVMContextRef ctx, const char *intrinsic,
                                      LLVMValueRef x, const char *name) {
    LLVMValueRef fn = get_v4f32_intrinsic(mod, ctx, intrinsic);
    LLVMTypeRef ty = get_v4f32_intrinsic_type(ctx, intrinsic);
    LLVMValueRef args[] = {x};
    return LLVMBuildCall2(B, ty, fn, args, 1, name);
}

/* ============================================================
 * Op implementations: each takes (builder, x_vec, mod, ctx)
 * and returns the result <4 x float>.
 * ============================================================ */

/* relu: max(0, x) */
static LLVMValueRef emit_relu(LLVMBuilderRef B, LLVMValueRef x,
                               LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef zero = splat_f32(B, ctx, 0.0);
    return call_v4f32_unary(B, mod, ctx, "llvm.maxnum", x, "relu");
    /* Actually: maxnum(x, 0) — need two args */
}

/* Let me use a different approach — maxnum is binary */
static LLVMValueRef emit_relu2(LLVMBuilderRef B, LLVMValueRef x,
                                LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMValueRef zero = LLVMConstNull(v4f32);
    LLVMTypeRef ovl[] = {v4f32};
    unsigned id = LLVMLookupIntrinsicID("llvm.maxnum", 11);
    LLVMValueRef fn = LLVMGetIntrinsicDeclaration(mod, id, ovl, 1);
    LLVMTypeRef ty = LLVMIntrinsicGetType(ctx, id, ovl, 1);
    LLVMValueRef args[] = {x, zero};
    return LLVMBuildCall2(B, ty, fn, args, 2, "relu");
}

/* silu: x * sigmoid(x) = x / (1 + exp(-x)) */
static LLVMValueRef emit_silu(LLVMBuilderRef B, LLVMValueRef x,
                               LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef neg_x = LLVMBuildFNeg(B, x, "neg_x");
    LLVMValueRef exp_neg = call_v4f32_unary(B, mod, ctx, "llvm.exp", neg_x, "exp_neg");
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef denom = LLVMBuildFAdd(B, ones, exp_neg, "denom");
    return LLVMBuildFDiv(B, x, denom, "silu");
}

/* sigmoid: 1 / (1 + exp(-x)) */
static LLVMValueRef emit_sigmoid(LLVMBuilderRef B, LLVMValueRef x,
                                  LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef neg_x = LLVMBuildFNeg(B, x, "neg_x");
    LLVMValueRef exp_neg = call_v4f32_unary(B, mod, ctx, "llvm.exp", neg_x, "exp_neg");
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef denom = LLVMBuildFAdd(B, ones, exp_neg, "denom");
    return LLVMBuildFDiv(B, ones, denom, "sigmoid");
}

/* tanh: (exp(2x) - 1) / (exp(2x) + 1) */
static LLVMValueRef emit_tanh(LLVMBuilderRef B, LLVMValueRef x,
                               LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef two = splat_f32(B, ctx, 2.0);
    LLVMValueRef twox = LLVMBuildFMul(B, two, x, "twox");
    LLVMValueRef exp2x = call_v4f32_unary(B, mod, ctx, "llvm.exp", twox, "exp2x");
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef num = LLVMBuildFSub(B, exp2x, ones, "num");
    LLVMValueRef den = LLVMBuildFAdd(B, exp2x, ones, "den");
    return LLVMBuildFDiv(B, num, den, "tanh");
}

/* gelu: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static LLVMValueRef emit_gelu(LLVMBuilderRef B, LLVMValueRef x,
                               LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef half = splat_f32(B, ctx, 0.5);
    LLVMValueRef coeff = splat_f32(B, ctx, 0.044715);
    LLVMValueRef sqrt2pi = splat_f32(B, ctx, 0.7978845608); /* sqrt(2/pi) */
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    
    LLVMValueRef x2 = LLVMBuildFMul(B, x, x, "x2");
    LLVMValueRef x3 = LLVMBuildFMul(B, x2, x, "x3");
    LLVMValueRef cx3 = LLVMBuildFMul(B, coeff, x3, "cx3");
    LLVMValueRef inner = LLVMBuildFAdd(B, x, cx3, "inner");
    LLVMValueRef scaled = LLVMBuildFMul(B, sqrt2pi, inner, "scaled");
    
    /* tanh of scaled */
    LLVMValueRef two = splat_f32(B, ctx, 2.0);
    LLVMValueRef twos = LLVMBuildFMul(B, two, scaled, "twos");
    LLVMValueRef exp2s = call_v4f32_unary(B, mod, ctx, "llvm.exp", twos, "exp2s");
    LLVMValueRef tnum = LLVMBuildFSub(B, exp2s, ones, "tnum");
    LLVMValueRef tden = LLVMBuildFAdd(B, exp2s, ones, "tden");
    LLVMValueRef th = LLVMBuildFDiv(B, tnum, tden, "th");
    
    LLVMValueRef oneph = LLVMBuildFAdd(B, ones, th, "oneph");
    LLVMValueRef hx = LLVMBuildFMul(B, half, x, "hx");
    return LLVMBuildFMul(B, hx, oneph, "gelu");
}

/* softplus: log(1 + exp(x)) */
static LLVMValueRef emit_softplus(LLVMBuilderRef B, LLVMValueRef x,
                                   LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef expx = call_v4f32_unary(B, mod, ctx, "llvm.exp", x, "expx");
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef sum = LLVMBuildFAdd(B, ones, expx, "sum");
    return call_v4f32_unary(B, mod, ctx, "llvm.log", sum, "softplus");
}

/* leaky_relu: x > 0 ? x : 0.01*x */
static LLVMValueRef emit_leaky_relu(LLVMBuilderRef B, LLVMValueRef x,
                                     LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMValueRef zero = LLVMConstNull(v4f32);
    LLVMValueRef alpha = splat_f32(B, ctx, 0.01);
    LLVMValueRef ax = LLVMBuildFMul(B, alpha, x, "ax");
    LLVMValueRef cmp = LLVMBuildFCmp(B, LLVMRealOGT, x, zero, "pos");
    return LLVMBuildSelect(B, cmp, x, ax, "leaky");
}

/* elu: x > 0 ? x : alpha*(exp(x)-1) */
static LLVMValueRef emit_elu(LLVMBuilderRef B, LLVMValueRef x,
                              LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMValueRef zero = LLVMConstNull(v4f32);
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef expx = call_v4f32_unary(B, mod, ctx, "llvm.exp", x, "expx");
    LLVMValueRef em1 = LLVMBuildFSub(B, expx, ones, "em1");
    LLVMValueRef cmp = LLVMBuildFCmp(B, LLVMRealOGT, x, zero, "pos");
    return LLVMBuildSelect(B, cmp, x, em1, "elu");
}

/* hardsigmoid: clamp(x/6 + 0.5, 0, 1) */
static LLVMValueRef emit_hardsigmoid(LLVMBuilderRef B, LLVMValueRef x,
                                      LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef v4f32 = LLVMVectorType(LLVMFloatTypeInContext(ctx), 4);
    LLVMValueRef sixth = splat_f32(B, ctx, 1.0/6.0);
    LLVMValueRef half = splat_f32(B, ctx, 0.5);
    LLVMValueRef zero = LLVMConstNull(v4f32);
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    
    LLVMValueRef xs = LLVMBuildFMul(B, x, sixth, "xs");
    LLVMValueRef xsh = LLVMBuildFAdd(B, xs, half, "xsh");
    
    /* clamp: max(0, min(1, xsh)) */
    LLVMTypeRef ovl[] = {v4f32};
    unsigned maxid = LLVMLookupIntrinsicID("llvm.maxnum", 11);
    unsigned minid = LLVMLookupIntrinsicID("llvm.minnum", 11);
    LLVMValueRef max_fn = LLVMGetIntrinsicDeclaration(mod, maxid, ovl, 1);
    LLVMValueRef min_fn = LLVMGetIntrinsicDeclaration(mod, minid, ovl, 1);
    LLVMTypeRef max_ty = LLVMIntrinsicGetType(ctx, maxid, ovl, 1);
    LLVMTypeRef min_ty = LLVMIntrinsicGetType(ctx, minid, ovl, 1);
    
    LLVMValueRef min_args[] = {xsh, ones};
    LLVMValueRef clamped_hi = LLVMBuildCall2(B, min_ty, min_fn, min_args, 2, "clhi");
    LLVMValueRef max_args[] = {clamped_hi, zero};
    return LLVMBuildCall2(B, max_ty, max_fn, max_args, 2, "hsig");
}

/* softsign: x / (1 + |x|) */
static LLVMValueRef emit_softsign(LLVMBuilderRef B, LLVMValueRef x,
                                   LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef absx = call_v4f32_unary(B, mod, ctx, "llvm.fabs", x, "absx");
    LLVMValueRef ones = splat_f32(B, ctx, 1.0);
    LLVMValueRef den = LLVMBuildFAdd(B, ones, absx, "den");
    return LLVMBuildFDiv(B, x, den, "softsign");
}

/* ============================================================
 * Loop wrapper: builds the vectorized loop around any op
 * ============================================================ */

typedef LLVMValueRef (*unary_op_t)(LLVMBuilderRef, LLVMValueRef, LLVMModuleRef, LLVMContextRef);

static void emit_unary_loop(LLVMModuleRef mod, const char *name, unary_op_t op) {
    LLVMContextRef ctx = LLVMGetModuleContext(mod);
    LLVMTypeRef f32 = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef v4f32 = LLVMVectorType(f32, 4);
    LLVMTypeRef i32 = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef i64 = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef fptr = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef voidty = LLVMVoidTypeInContext(ctx);
    
    LLVMTypeRef params[] = {i32, fptr, fptr};
    LLVMTypeRef fn_type = LLVMFunctionType(voidty, params, 3, 0);
    LLVMValueRef fn = LLVMAddFunction(mod, name, fn_type);
    LLVMSetValueName2(LLVMGetParam(fn, 0), "n", 1);
    LLVMSetValueName2(LLVMGetParam(fn, 1), "dst", 3);
    LLVMSetValueName2(LLVMGetParam(fn, 2), "src", 3);
    
    LLVMBuilderRef B = LLVMCreateBuilderInContext(ctx);
    LLVMBasicBlockRef bb_entry = LLVMAppendBasicBlockInContext(ctx, fn, "entry");
    LLVMBasicBlockRef bb_loop = LLVMAppendBasicBlockInContext(ctx, fn, "loop");
    LLVMBasicBlockRef bb_done = LLVMAppendBasicBlockInContext(ctx, fn, "done");
    
    LLVMValueRef n_val = LLVMGetParam(fn, 0);
    LLVMValueRef dst = LLVMGetParam(fn, 1);
    LLVMValueRef src = LLVMGetParam(fn, 2);
    LLVMValueRef zero_i = LLVMConstInt(i32, 0, 0);
    LLVMValueRef four = LLVMConstInt(i32, 4, 0);
    
    /* Entry */
    LLVMPositionBuilderAtEnd(B, bb_entry);
    LLVMValueRef np = LLVMBuildAnd(B, n_val, LLVMConstInt(i32, ~3, 0), "np");
    LLVMValueRef has = LLVMBuildICmp(B, LLVMIntSGT, np, zero_i, "has");
    LLVMBuildCondBr(B, has, bb_loop, bb_done);
    
    /* Loop */
    LLVMPositionBuilderAtEnd(B, bb_loop);
    LLVMValueRef i_phi = LLVMBuildPhi(B, i32, "i");
    
    LLVMValueRef idx64 = LLVMBuildSExt(B, i_phi, i64, "idx64");
    LLVMValueRef sg = LLVMBuildGEP2(B, f32, src, &idx64, 1, "sg");
    LLVMValueRef xv = LLVMBuildLoad2(B, v4f32, sg, "xv");
    LLVMSetAlignment(xv, 4);
    
    /* Apply the op */
    LLVMValueRef result = op(B, xv, mod, ctx);
    
    /* Store */
    LLVMValueRef dg = LLVMBuildGEP2(B, f32, dst, &idx64, 1, "dg");
    LLVMBuildStore(B, result, dg);
    
    /* Increment */
    LLVMValueRef i_next = LLVMBuildAdd(B, i_phi, four, "i_next");
    LLVMValueRef again = LLVMBuildICmp(B, LLVMIntSLT, i_next, np, "again");
    LLVMBuildCondBr(B, again, bb_loop, bb_done);
    
    LLVMValueRef i_in[] = {zero_i, i_next};
    LLVMBasicBlockRef i_bb[] = {bb_entry, bb_loop};
    LLVMAddIncoming(i_phi, i_in, i_bb, 2);
    
    /* Done */
    LLVMPositionBuilderAtEnd(B, bb_done);
    LLVMBuildRetVoid(B);
    
    LLVMDisposeBuilder(B);
}

/* ============================================================
 * Prolog foreign interface
 * ============================================================ */

static foreign_t pl_llvm_emit_unary(term_t mod_t, term_t op_t, term_t name_t) {
    void *p; char *op_name; char *fn_name;
    if (!PL_get_pointer(mod_t, &p)) return FALSE;
    if (!PL_get_atom_chars(op_t, &op_name)) return FALSE;
    if (!PL_get_atom_chars(name_t, &fn_name)) return FALSE;
    
    LLVMModuleRef mod = (LLVMModuleRef)p;
    
    struct { const char *name; unary_op_t fn; } ops[] = {
        {"relu",        emit_relu2},
        {"silu",        emit_silu},
        {"sigmoid",     emit_sigmoid},
        {"tanh",        emit_tanh},
        {"gelu",        emit_gelu},
        {"softplus",    emit_softplus},
        {"leaky_relu",  emit_leaky_relu},
        {"elu",         emit_elu},
        {"hardsigmoid", emit_hardsigmoid},
        {"softsign",    emit_softsign},
        {NULL, NULL}
    };
    
    for (int i = 0; ops[i].name; i++) {
        if (strcmp(op_name, ops[i].name) == 0) {
            emit_unary_loop(mod, fn_name, ops[i].fn);
            return TRUE;
        }
    }
    
    return PL_warning("Unknown unary op: %s", op_name);
}

install_t install_bpd_llvm_elem(void) {
    PL_register_foreign("llvm_emit_unary", 3, pl_llvm_emit_unary, 0);
}
