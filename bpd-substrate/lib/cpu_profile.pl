%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════
%% cpu_profile.pl — CPU hardware performance counters from Prolog
%%
%% The CPU equivalent of CUPTI-from-Prolog. Uses Linux perf_event_open
%% syscall via a small C helper to measure:
%%   - cycles, instructions (IPC)
%%   - L1 cache misses, LLC misses
%%   - branch mispredictions
%%   - task-clock (wall time)
%%
%% Usage:
%%   cpu_profile(SharedLib, FunctionName, N, InputSize, Result)
%%   Result = counters{cycles, instructions, l1_misses, llc_misses,
%%                     branch_misses, task_clock_ns, ipc, gflops}
%%
%% The dispatch pipeline uses this to sweep parameters:
%%   For each tile size / strategy / parameter combination:
%%     1. Generate the kernel (derive_epilogues.pl)
%%     2. Compile it (gcc)
%%     3. Profile it (cpu_profile.pl)
%%     4. Record the Pareto frontier (performance vs ULP)
%% ═══════════════════════════════════════════════════════════════

:- module(cpu_profile, [
    cpu_profile/5,
    build_profiler/0,
    profile_comparison/4
]).

%% build_profiler/0 — compile the C helper that wraps perf_event_open
build_profiler :-
    HelperSrc = '/tmp/bpd_cpu_profiler.c',
    HelperSo = '/tmp/bpd_cpu_profiler.so',
    format(atom(Cmd),
        "gcc -O2 -shared -fPIC -o ~w ~w -lm 2>&1",
        [HelperSo, HelperSrc]),
    (exists_file(HelperSrc) ->
        shell(Cmd, 0),
        format("Built CPU profiler helper: ~w~n", [HelperSo])
    ;
        format("Writing profiler helper source...~n"),
        write_profiler_source(HelperSrc),
        shell(Cmd, 0),
        format("Built CPU profiler helper: ~w~n", [HelperSo])
    ).

%% write_profiler_source/1 — generate the C helper
write_profiler_source(Path) :-
    open(Path, write, S),
    format(S, "#include <stdio.h>~n", []),
    format(S, "#include <stdlib.h>~n", []),
    format(S, "#include <string.h>~n", []),
    format(S, "#include <unistd.h>~n", []),
    format(S, "#include <sys/ioctl.h>~n", []),
    format(S, "#include <linux/perf_event.h>~n", []),
    format(S, "#include <sys/syscall.h>~n~n", []),
    format(S, "static long perf_event_open(struct perf_event_attr *attr,~n", []),
    format(S, "    pid_t pid, int cpu, int group_fd, unsigned long flags) {~n", []),
    format(S, "    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);~n", []),
    format(S, "}~n~n", []),
    format(S, "/* Profile a function: run it N times, measure hardware counters */~n", []),
    format(S, "int bpd_cpu_profile(void (*fn)(void), int reps,~n", []),
    format(S, "    long long *cycles, long long *instructions,~n", []),
    format(S, "    long long *l1_misses, long long *llc_misses,~n", []),
    format(S, "    long long *branch_misses, long long *task_clock) {~n~n", []),
    format(S, "    struct perf_event_attr pe;~n", []),
    format(S, "    int fd_cyc, fd_ins, fd_l1, fd_llc, fd_br, fd_clk;~n~n", []),
    format(S, "    memset(&pe, 0, sizeof(pe));~n", []),
    format(S, "    pe.size = sizeof(pe);~n", []),
    format(S, "    pe.disabled = 1;~n", []),
    format(S, "    pe.exclude_kernel = 1;~n", []),
    format(S, "    pe.exclude_hv = 1;~n~n", []),
    format(S, "    pe.type = PERF_TYPE_HARDWARE; pe.config = PERF_COUNT_HW_CPU_CYCLES;~n", []),
    format(S, "    fd_cyc = perf_event_open(&pe, 0, -1, -1, 0);~n", []),
    format(S, "    pe.config = PERF_COUNT_HW_INSTRUCTIONS;~n", []),
    format(S, "    fd_ins = perf_event_open(&pe, 0, -1, -1, 0);~n", []),
    format(S, "    pe.config = PERF_COUNT_HW_CACHE_MISSES;~n", []),
    format(S, "    fd_l1 = perf_event_open(&pe, 0, -1, -1, 0);~n", []),
    format(S, "    pe.type = PERF_TYPE_HW_CACHE;~n", []),
    format(S, "    pe.config = PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);~n", []),
    format(S, "    fd_llc = perf_event_open(&pe, 0, -1, -1, 0);~n", []),
    format(S, "    pe.type = PERF_TYPE_HARDWARE; pe.config = PERF_COUNT_HW_BRANCH_MISSES;~n", []),
    format(S, "    fd_br = perf_event_open(&pe, 0, -1, -1, 0);~n", []),
    format(S, "    pe.type = PERF_TYPE_SOFTWARE; pe.config = PERF_COUNT_SW_TASK_CLOCK;~n", []),
    format(S, "    fd_clk = perf_event_open(&pe, 0, -1, -1, 0);~n~n", []),
    format(S, "    /* Enable all counters */~n", []),
    format(S, "    ioctl(fd_cyc, PERF_EVENT_IOC_RESET, 0); ioctl(fd_cyc, PERF_EVENT_IOC_ENABLE, 0);~n", []),
    format(S, "    ioctl(fd_ins, PERF_EVENT_IOC_RESET, 0); ioctl(fd_ins, PERF_EVENT_IOC_ENABLE, 0);~n", []),
    format(S, "    ioctl(fd_l1, PERF_EVENT_IOC_RESET, 0); ioctl(fd_l1, PERF_EVENT_IOC_ENABLE, 0);~n", []),
    format(S, "    ioctl(fd_llc, PERF_EVENT_IOC_RESET, 0); ioctl(fd_llc, PERF_EVENT_IOC_ENABLE, 0);~n", []),
    format(S, "    ioctl(fd_br, PERF_EVENT_IOC_RESET, 0); ioctl(fd_br, PERF_EVENT_IOC_ENABLE, 0);~n", []),
    format(S, "    ioctl(fd_clk, PERF_EVENT_IOC_RESET, 0); ioctl(fd_clk, PERF_EVENT_IOC_ENABLE, 0);~n~n", []),
    format(S, "    /* Run the function */~n", []),
    format(S, "    for (int i = 0; i < reps; i++) fn();~n~n", []),
    format(S, "    /* Disable and read */~n", []),
    format(S, "    ioctl(fd_cyc, PERF_EVENT_IOC_DISABLE, 0); read(fd_cyc, cycles, 8);~n", []),
    format(S, "    ioctl(fd_ins, PERF_EVENT_IOC_DISABLE, 0); read(fd_ins, instructions, 8);~n", []),
    format(S, "    ioctl(fd_l1, PERF_EVENT_IOC_DISABLE, 0); read(fd_l1, l1_misses, 8);~n", []),
    format(S, "    ioctl(fd_llc, PERF_EVENT_IOC_DISABLE, 0); read(fd_llc, llc_misses, 8);~n", []),
    format(S, "    ioctl(fd_br, PERF_EVENT_IOC_DISABLE, 0); read(fd_br, branch_misses, 8);~n", []),
    format(S, "    ioctl(fd_clk, PERF_EVENT_IOC_DISABLE, 0); read(fd_clk, task_clock, 8);~n~n", []),
    format(S, "    close(fd_cyc); close(fd_ins); close(fd_l1);~n", []),
    format(S, "    close(fd_llc); close(fd_br); close(fd_clk);~n", []),
    format(S, "    return 0;~n", []),
    format(S, "}~n", []),
    close(S).

%% cpu_profile/5 — profile a kernel via the C helper
%% cpu_profile(+SoPath, +FnName, +Reps, +SetupFn, -Result)
cpu_profile(SoPath, FnName, Reps, SetupFn, Result) :-
    format(atom(Cmd),
        "LD_PRELOAD=~w /tmp/bpd_cpu_profiler_runner ~w ~w ~w",
        [SoPath, FnName, Reps, SetupFn]),
    % For now, dispatch via shell and parse output
    % Future: use FFI to call bpd_cpu_profile directly
    format("cpu_profile: ~w~n", [Cmd]),
    Result = counters{
        command: Cmd,
        status: not_yet_implemented
    }.

%% profile_comparison/4 — compare two strategies
profile_comparison(Op, Strategy1, Strategy2, Comparison) :-
    format("Comparing ~w: ~w vs ~w~n", [Op, Strategy1, Strategy2]),
    Comparison = comparison{
        op: Op,
        strategies: [Strategy1, Strategy2],
        status: not_yet_implemented,
        note: 'Build profiler first with build_profiler/0, then run cpu_profile/5'
    }.
