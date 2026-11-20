/* cuda_profiler.c — CUPTI Event/Metric API for SWI-Prolog FLI
 *
 * Extends cuda_launch.so with hardware performance counter support.
 * Queries CUPTI Event API on sm_61 (Pascal/P4) — bypasses the ncu
 * version limitation that dropped sm_61 support.
 *
 * Prolog predicates:
 *   cupti_list_events(-EventNames)        — enumerate available events
 *   cupti_list_metrics(-MetricNames)      — enumerate available metrics
 *   cupti_event_group_create(-GroupHandle) — create event group
 *   cupti_event_group_add(+Group, +EventName) — add event to group
 *   cupti_event_group_enable(+Group)      — enable collection
 *   cupti_event_group_disable(+Group)     — disable collection
 *   cupti_event_group_read(+Group, +EventName, -Value) — read counter
 *   cupti_event_group_destroy(+Group)     — free group
 *   cupti_metric_create(+MetricName, -MetricHandle) — prepare metric
 *   cupti_metric_evaluate(+MetricHandle, +EventGroupSets, -Value) — compute
 *
 * Build (append to cuda_launch.so):
 *   gcc -shared -fPIC -O2 cuda_profiler.c -o cuda_profiler.so \
 *       -I<swipl_include> -I<cupti_include> -L<cupti_lib> -lcupti -lcuda
 *
 * Author: mavchin, Ruach Tov Collective, October 2026.
 * Extends cuda_launch.c (May 2026).
 */

#include <SWI-Prolog.h>
#include <cupti.h>
#include <cupti_events.h>
#include <cuda.h>
#include <string.h>
#include <stdlib.h>

/* ── Error checking ── */
#define CUPTI_CHECK(call) do { \
    CUptiResult _r = (call); \
    if (_r != CUPTI_SUCCESS) { \
        const char *_msg; \
        cuptiGetResultString(_r, &_msg); \
        return PL_warning("CUPTI error: %s", _msg); \
    } \
} while(0)

/* ── List available events for device 0 ── */
static foreign_t pl_cupti_list_events(term_t list_t) {
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    uint32_t numDomains;
    cuptiDeviceGetNumEventDomains(dev, &numDomains);

    CUpti_EventDomainID *domainIds = malloc(numDomains * sizeof(CUpti_EventDomainID));
    size_t sz = numDomains * sizeof(CUpti_EventDomainID);
    cuptiDeviceEnumEventDomains(dev, &sz, domainIds);

    term_t tail = PL_copy_term_ref(list_t);
    term_t head = PL_new_term_ref();

    for (uint32_t d = 0; d < numDomains; d++) {
        uint32_t numEvents;
        cuptiEventDomainGetNumEvents(domainIds[d], &numEvents);

        CUpti_EventID *eventIds = malloc(numEvents * sizeof(CUpti_EventID));
        size_t esz = numEvents * sizeof(CUpti_EventID);
        cuptiEventDomainEnumEvents(domainIds[d], &esz, eventIds);

        for (uint32_t e = 0; e < numEvents; e++) {
            char name[256];
            size_t nsz = sizeof(name);
            cuptiEventGetAttribute(eventIds[e], CUPTI_EVENT_ATTR_NAME, &nsz, name);

            if (!PL_unify_list(tail, head, tail) ||
                !PL_unify_atom_chars(head, name)) {
                free(eventIds);
                free(domainIds);
                PL_fail;
            }
        }
        free(eventIds);
    }
    free(domainIds);
    return PL_unify_nil(tail);
}

/* ── Create event group ── */
static foreign_t pl_cupti_event_group_create(term_t group_t) {
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);

    CUpti_EventGroup group;
    CUPTI_CHECK(cuptiEventGroupCreate(ctx, &group, 0));

    return PL_unify_int64(group_t, (int64_t)group);
}

/* ── Add event to group ── */
static foreign_t pl_cupti_event_group_add(term_t group_t, term_t name_t) {
    int64_t group_i;
    char *name;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    if (!PL_get_atom_chars(name_t, &name)) PL_fail;

    CUpti_EventGroup group = (CUpti_EventGroup)group_i;

    /* Look up event ID by name */
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    /* Enumerate all events to find the named one */
    uint32_t numDomains;
    cuptiDeviceGetNumEventDomains(dev, &numDomains);
    CUpti_EventDomainID *domainIds = malloc(numDomains * sizeof(CUpti_EventDomainID));
    size_t sz = numDomains * sizeof(CUpti_EventDomainID);
    cuptiDeviceEnumEventDomains(dev, &sz, domainIds);

    CUpti_EventID found = 0;
    int ok = 0;
    for (uint32_t d = 0; d < numDomains && !ok; d++) {
        uint32_t numEvents;
        cuptiEventDomainGetNumEvents(domainIds[d], &numEvents);
        CUpti_EventID *eventIds = malloc(numEvents * sizeof(CUpti_EventID));
        size_t esz = numEvents * sizeof(CUpti_EventID);
        cuptiEventDomainEnumEvents(domainIds[d], &esz, eventIds);

        for (uint32_t e = 0; e < numEvents; e++) {
            char ename[256];
            size_t nsz = sizeof(ename);
            cuptiEventGetAttribute(eventIds[e], CUPTI_EVENT_ATTR_NAME, &nsz, ename);
            if (strcmp(ename, name) == 0) {
                found = eventIds[e];
                ok = 1;
                break;
            }
        }
        free(eventIds);
    }
    free(domainIds);

    if (!ok) return PL_warning("CUPTI event '%s' not found", name);

    CUPTI_CHECK(cuptiEventGroupAddEvent(group, found));
    PL_succeed;
}

/* ── Enable/Disable event group ── */
static foreign_t pl_cupti_event_group_enable(term_t group_t) {
    int64_t group_i;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    CUPTI_CHECK(cuptiEventGroupEnable((CUpti_EventGroup)group_i));
    PL_succeed;
}

static foreign_t pl_cupti_event_group_disable(term_t group_t) {
    int64_t group_i;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    CUPTI_CHECK(cuptiEventGroupDisable((CUpti_EventGroup)group_i));
    PL_succeed;
}

/* ── Read event counter value ── */
static foreign_t pl_cupti_event_group_read(term_t group_t, term_t name_t, term_t value_t) {
    int64_t group_i;
    char *name;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    if (!PL_get_atom_chars(name_t, &name)) PL_fail;

    CUpti_EventGroup group = (CUpti_EventGroup)group_i;

    /* Find event ID by name (same lookup as add) */
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    uint32_t numDomains;
    cuptiDeviceGetNumEventDomains(dev, &numDomains);
    CUpti_EventDomainID *domainIds = malloc(numDomains * sizeof(CUpti_EventDomainID));
    size_t sz = numDomains * sizeof(CUpti_EventDomainID);
    cuptiDeviceEnumEventDomains(dev, &sz, domainIds);

    CUpti_EventID found = 0;
    int ok = 0;
    for (uint32_t d = 0; d < numDomains && !ok; d++) {
        uint32_t numEvents;
        cuptiEventDomainGetNumEvents(domainIds[d], &numEvents);
        CUpti_EventID *eventIds = malloc(numEvents * sizeof(CUpti_EventID));
        size_t esz = numEvents * sizeof(CUpti_EventID);
        cuptiEventDomainEnumEvents(domainIds[d], &esz, eventIds);
        for (uint32_t e = 0; e < numEvents; e++) {
            char ename[256];
            size_t nsz = sizeof(ename);
            cuptiEventGetAttribute(eventIds[e], CUPTI_EVENT_ATTR_NAME, &nsz, ename);
            if (strcmp(ename, name) == 0) { found = eventIds[e]; ok = 1; break; }
        }
        free(eventIds);
    }
    free(domainIds);
    if (!ok) return PL_warning("CUPTI event '%s' not found", name);

    /* Read the counter */
    uint64_t value;
    size_t vsz = sizeof(value);
    size_t idsz = sizeof(found);
    CUPTI_CHECK(cuptiEventGroupReadEvent(group,
        CUPTI_EVENT_READ_FLAG_FORCE_INT, found, &vsz, &value));

    return PL_unify_int64(value_t, (int64_t)value);
}

/* ── Destroy event group ── */
static foreign_t pl_cupti_event_group_destroy(term_t group_t) {
    int64_t group_i;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    CUPTI_CHECK(cuptiEventGroupDestroy((CUpti_EventGroup)group_i));
    PL_succeed;
}

/* ── Reset event group counters ── */
static foreign_t pl_cupti_event_group_reset(term_t group_t) {
    int64_t group_i;
    if (!PL_get_int64(group_t, &group_i)) PL_fail;
    CUPTI_CHECK(cuptiEventGroupResetAllEvents((CUpti_EventGroup)group_i));
    PL_succeed;
}

/* ── Registration ── */
install_t install_cuda_profiler(void) {
    PL_register_foreign("cupti_list_events",        1, pl_cupti_list_events,        0);
    PL_register_foreign("cupti_event_group_create",  1, pl_cupti_event_group_create,  0);
    PL_register_foreign("cupti_event_group_add",     2, pl_cupti_event_group_add,     0);
    PL_register_foreign("cupti_event_group_enable",  1, pl_cupti_event_group_enable,  0);
    PL_register_foreign("cupti_event_group_disable", 1, pl_cupti_event_group_disable, 0);
    PL_register_foreign("cupti_event_group_read",    3, pl_cupti_event_group_read,    0);
    PL_register_foreign("cupti_event_group_reset",   1, pl_cupti_event_group_reset,   0);
    PL_register_foreign("cupti_event_group_destroy", 1, pl_cupti_event_group_destroy, 0);
}
