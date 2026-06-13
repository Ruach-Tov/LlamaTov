// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/* cupti_bridge.c — SWI-Prolog Foreign Language Interface for CUPTI profiling
 *
 * Exposes CUPTI PC sampling as native Prolog predicates via PLF.
 * No ctypes, no Python, no intermediate layers.
 *
 * Predicates:
 *   cupti_init/0           — initialize PC sampling
 *   cupti_shutdown/0       — disable profiling
 *   cupti_reset/0          — clear counters
 *   cupti_flush/0          — flush activity buffers
 *   cupti_stall_report/1   — get stall data as a Prolog list
 *   cupti_suggest/1        — get optimization suggestions as a list
 *
 * Build: swipl-ld -shared -o cupti_bridge lib/cupti_bridge.c lib/bpd_cupti_profile.c \
 *          -lcupti -lcuda -I$CUPTI_INC -L$CUPTI_LIB
 *
 * Usage in Prolog:
 *   :- use_foreign_library(cupti_bridge).
 *   ?- cupti_init, run_kernel(...), cupti_flush, cupti_stall_report(Stalls).
 *   Stalls = [none-3.6, inst_fetch-3.2, exec_dep-14.4, mem_dep-61.8, ...]
 */

#include <SWI-Prolog.h>
#include <string.h>
#include <stdio.h>

/* Import from bpd_cupti_profile.c */
typedef struct {
    unsigned long long none, inst_fetch, exec_dependency, memory_dependency;
    unsigned long long texture, sync, constant_memory, pipe_busy;
    unsigned long long memory_throttle, not_selected, other, sleeping;
    unsigned long long total_samples;
} stall_counters_t;

extern int bpd_cupti_init(void);
extern int bpd_cupti_flush(void);
extern int bpd_cupti_get_stalls(stall_counters_t* out);
extern void bpd_cupti_reset(void);
extern int bpd_cupti_shutdown(void);

/* ================================================================
 * Helper: build a Prolog list of Key-Value pairs from stall data
 * ================================================================ */
static int build_stall_list(term_t list, stall_counters_t* s) {
    term_t l = PL_copy_term_ref(list);
    term_t head = PL_new_term_ref();
    
    if (s->total_samples == 0) {
        return PL_unify_nil(l);
    }
    
    double total = (double)s->total_samples;
    
    /* Helper macro: add StallName-Percentage to the list */
    #define ADD_STALL(name_str, field) do { \
        double pct = 100.0 * s->field / total; \
        if (pct > 0.05) { \
            term_t pair = PL_new_term_ref(); \
            term_t key = PL_new_term_ref(); \
            term_t val = PL_new_term_ref(); \
            PL_put_atom_chars(key, name_str); \
            PL_put_float(val, pct); \
            if (!PL_cons_functor(pair, PL_new_functor(PL_new_atom("-"), 2), key, val)) return FALSE; \
            if (!PL_unify_list(l, head, l)) return FALSE; \
            if (!PL_unify(head, pair)) return FALSE; \
        } \
    } while(0)
    
    ADD_STALL("none", none);
    ADD_STALL("inst_fetch", inst_fetch);
    ADD_STALL("exec_dependency", exec_dependency);
    ADD_STALL("memory_dependency", memory_dependency);
    ADD_STALL("texture", texture);
    ADD_STALL("sync", sync);
    ADD_STALL("constant_memory", constant_memory);
    ADD_STALL("pipe_busy", pipe_busy);
    ADD_STALL("memory_throttle", memory_throttle);
    ADD_STALL("not_selected", not_selected);
    ADD_STALL("sleeping", sleeping);
    ADD_STALL("other", other);
    
    #undef ADD_STALL
    
    return PL_unify_nil(l);
}

/* ================================================================
 * PLF Predicates
 * ================================================================ */

/* cupti_init/0 — initialize CUPTI PC sampling */
static foreign_t pl_cupti_init(void) {
    int rc = bpd_cupti_init();
    if (rc != 0) {
        return PL_warning("cupti_init failed (rc=%d)", rc);
    }
    return TRUE;
}

/* cupti_shutdown/0 — disable profiling */
static foreign_t pl_cupti_shutdown(void) {
    bpd_cupti_shutdown();
    return TRUE;
}

/* cupti_reset/0 — clear counters */
static foreign_t pl_cupti_reset(void) {
    bpd_cupti_reset();
    return TRUE;
}

/* cupti_flush/0 — flush activity buffers */
static foreign_t pl_cupti_flush(void) {
    int rc = bpd_cupti_flush();
    if (rc != 0) {
        return PL_warning("cupti_flush failed (rc=%d)", rc);
    }
    return TRUE;
}

/* cupti_stall_report(-StallList) — get stall data as Key-Percentage pairs */
static foreign_t pl_cupti_stall_report(term_t stall_list) {
    stall_counters_t stalls;
    bpd_cupti_get_stalls(&stalls);
    return build_stall_list(stall_list, &stalls);
}

/* cupti_total_samples(-N) — get total sample count */
static foreign_t pl_cupti_total_samples(term_t n) {
    stall_counters_t stalls;
    bpd_cupti_get_stalls(&stalls);
    return PL_unify_int64(n, stalls.total_samples);
}

/* cupti_suggest(-Suggestions) — get optimization suggestions as a list */
static foreign_t pl_cupti_suggest(term_t suggestions) {
    stall_counters_t stalls;
    bpd_cupti_get_stalls(&stalls);
    
    if (stalls.total_samples == 0) {
        return PL_unify_nil(suggestions);
    }
    
    double total = (double)stalls.total_samples;
    double mem_pct = 100.0 * stalls.memory_dependency / total;
    double exec_pct = 100.0 * stalls.exec_dependency / total;
    double sync_pct = 100.0 * stalls.sync / total;
    double throttle_pct = 100.0 * stalls.memory_throttle / total;
    double const_pct = 100.0 * stalls.constant_memory / total;
    
    term_t l = PL_copy_term_ref(suggestions);
    term_t head = PL_new_term_ref();
    
    /* Build suggestion terms: suggest(Strategy, StallType, Percentage) */
    #define ADD_SUGGEST(strategy, stall_type, pct) do { \
        term_t s = PL_new_term_ref(); \
        term_t a0 = PL_new_term_refs(3); \
        PL_put_atom_chars(a0, strategy); \
        PL_put_atom_chars(a0+1, stall_type); \
        PL_put_float(a0+2, pct); \
        PL_cons_functor_v(s, PL_new_functor(PL_new_atom("suggest"), 3), a0); \
        if (!PL_unify_list(l, head, l)) return FALSE; \
        if (!PL_unify(head, s)) return FALSE; \
    } while(0)
    
    if (mem_pct > 30) ADD_SUGGEST("warp_shuffle", "memory_dependency", mem_pct);
    else if (mem_pct > 15) ADD_SUGGEST("shared_memory", "memory_dependency", mem_pct);
    if (exec_pct > 20) ADD_SUGGEST("increase_ilp", "exec_dependency", exec_pct);
    if (sync_pct > 15) ADD_SUGGEST("reduce_barriers", "sync", sync_pct);
    if (throttle_pct > 10) ADD_SUGGEST("coalesce_memory", "memory_throttle", throttle_pct);
    if (const_pct > 10) ADD_SUGGEST("use_registers", "constant_memory", const_pct);
    
    if (mem_pct <= 15 && exec_pct <= 20 && sync_pct <= 15)
        ADD_SUGGEST("well_optimized", "none", 0.0);
    
    #undef ADD_SUGGEST
    
    return PL_unify_nil(l);
}

/* ================================================================
 * PLF Registration
 * ================================================================ */

install_t install_cupti_bridge(void) {
    PL_register_foreign("cupti_init",           0, pl_cupti_init,           0);
    PL_register_foreign("cupti_shutdown",       0, pl_cupti_shutdown,       0);
    PL_register_foreign("cupti_reset",          0, pl_cupti_reset,          0);
    PL_register_foreign("cupti_flush",          0, pl_cupti_flush,          0);
    PL_register_foreign("cupti_stall_report",   1, pl_cupti_stall_report,   0);
    PL_register_foreign("cupti_total_samples",  1, pl_cupti_total_samples,  0);
    PL_register_foreign("cupti_suggest",        1, pl_cupti_suggest,        0);
}
