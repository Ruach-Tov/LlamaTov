%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% test_kernelbench_l1_cuda.pl — Scope C nvcc validation harness for L1.
%%
%% Per Heath's directive (via mavchin's accept of Scope C):
%% Wrap the existing emission pipeline with nvcc compilation, catching
%% ANY syntax/type error in emitted CUDA that the unit tests missed.
%%
%% Per medayek's robust-kbench finding:
%% Beyond syntactic correctness, future Scope C-Extended will add
%% semantic-correctness checks (compare against PyTorch reference) to
%% avoid the "exploitable loopholes" identified in the original
%% KernelBench. This commit lands the compile-check foundation.
%%
%% Target architecture: sm_61 (Tesla P4 — mavchin's GPU). This is the
%% hardware where the substrate's "scales DOWN as well as up" thesis
%% lives (per boneh's framing). Compiles for sm_61 means the emission
%% is compatible with PyTorch-abandoned Pascal-era GPUs.
%%
%% Running this test requires nvcc:
%%   NIXPKGS_ALLOW_UNFREE=1 nix-shell \
%%     -p swiProlog cudaPackages.cuda_nvcc cudaPackages.cuda_cudart \
%%     --impure --run "swipl -q test_kernelbench_l1_cuda.pl"
%%
%% On a node without nvcc, this test SKIPS — substrate-honest design:
%% don't fail just because nvcc isn't available. The harness reports
%% "SKIPPED (nvcc not in PATH)" and exits 0.

:- set_prolog_flag(double_quotes, codes).
:- use_module('../lib/c_ast').
:- use_module('../lib/kernel_templates').

%% Architecture target for compilation

%% cuda_include_flag(-Flag): the -I for CUDA headers. nvcc sometimes can't find cuda_runtime.h on a
%% bare PATH (it relies on the toolkit layout). Try $CUDA_INCLUDE, then $CUDA_HOME/include, then
%% derive from `which nvcc` (bin/../include). Empty if none found (nvcc's own default path is used).
:- dynamic toolchain_fact/2.

%% toolchain_include(-Inc): read the include dir from the SHARED toolchain facts (bpd/lib/toolchain.py
%% via write_prolog_facts -> /tmp/toolchain_facts.pl) — the single source of CUDA-env truth. Prefer it
%% so the Prolog harness and production learn env TOGETHER. Falls through if absent (substrate-honest).
toolchain_include(Inc) :-
    exists_file('/tmp/toolchain_facts.pl'),
    catch(consult('/tmp/toolchain_facts.pl'), _, fail),
    toolchain_fact('cuda_include', Inc).

cuda_include_flag(Flag) :-
    ( toolchain_include(TInc), TInc \= ''
    -> format(atom(Flag), '-I~w', [TInc])
    ;  getenv('CUDA_INCLUDE', Inc), Inc \= ''
    -> format(atom(Flag), '-I~w', [Inc])
    ;  getenv('CUDA_HOME', Home), Home \= '',
       atom_concat(Home, '/include', Inc)
    -> format(atom(Flag), '-I~w', [Inc])
    %% nix split-output fallback: nvcc's own package has no include/; the
    %% merged toolkit (same default as fact_dispatch.py) does. Check it
    %% BEFORE the derive-from-nvcc heuristic, which fails on nix.
    ;  Merged = '/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8/include',
       exists_directory(Merged)
    -> format(atom(Flag), '-I~w', [Merged])
    ;  catch(( setup_call_cleanup(
                 process_create(path(sh), ['-c', 'dirname $(dirname $(readlink -f $(which nvcc)))'],
                                [stdout(pipe(O)), process(P)]),
                 ( read_string(O, _, S0), process_wait(P, _) ),
                 close(O)),
               atom_string(Root, S0)), _, fail),
       split_string(Root, "\n", "", [RootS|_]), RootS \= "",
       format(atom(Flag), '-I~w/include', [RootS])
    -> true
    ;  Flag = '' ).

target_arch('sm_61').

%% nvcc_bin(-Bin): the FULL-toolkit nvcc. The bare PATH nvcc on nix is the
%% split cuda_nvcc package, which lacks nvvm/cicc (exit 127 mid-compile).
%% Use CUDA_HOME if set, else the canonical merged toolkit (same default as
%% bpd/lib/fact_dispatch.py). Falls back to PATH nvcc only if neither exists.
nvcc_bin(Bin) :-
    ( getenv('CUDA_HOME', H), H \= '',
      format(atom(B0), '~w/bin/nvcc', [H]), exists_file(B0)
    -> Bin = B0
    ;  B1 = '/nix/store/3y4mvymhwmnfi5d0vwyzcw7f7sqnqnkd-cuda-merged-12.8/bin/nvcc',
       exists_file(B1)
    -> Bin = B1
    ;  Bin = nvcc ).

%% Temporary file naming
%% per-user temp dir so multiple agents can run the harness concurrently (was a hardcoded shared
%% '/tmp/l1_cuda_validation' owned by the first runner -> Permission denied for everyone else).
tmp_dir(Dir) :-
    ( getenv('USER', U), U \= '' -> true ; U = 'anon' ),
    format(atom(Dir), '/tmp/l1_cuda_validation_~w', [U]).

run_tests :-
    %% First check if nvcc is available
    ( has_nvcc
    -> format("~nnvcc detected; running full validation harness~n", []),
       run_validation
    ;  format("~nnvcc NOT in PATH — skipping compile validation~n", []),
       format("To run full Scope C: NIXPKGS_ALLOW_UNFREE=1 nix-shell \\~n", []),
       format("    -p swiProlog cudaPackages.cuda_nvcc cudaPackages.cuda_cudart --impure \\~n", []),
       format("    --run 'swipl -q test_kernelbench_l1_cuda.pl'~n", []),
       run_structural_only
    ).

has_nvcc :-
    catch(
        ( shell('which nvcc > /dev/null 2>&1', 0) ),
        _,
        fail
    ).

%% ─────────────────────────────────────────────────────────────────────
%% Validation cases: one per kernel-template family + variant
%% ─────────────────────────────────────────────────────────────────────
%%
%% validation_case(+Name, +Goal)
%%   Name: human-readable label for the kernel variant
%%   Goal: a goal that produces a kernel AST in `K` (via succ_or_fail)
%%
%% Each case is a (label, generation goal) pair so that the harness
%% can emit, compile, and report uniformly.

%% Family 4: Conv (im2col) — 6 variants
validation_case('conv_2d_forward',           generate_kernel_im2col(ggml_conv_2d, 2, forward, _)).
validation_case('conv_1d_forward',           generate_kernel_im2col(ggml_conv_1d, 1, forward, _)).
validation_case('conv_3d_forward',           generate_kernel_im2col(ggml_conv_3d, 3, forward, _)).
validation_case('conv_transpose_1d',         generate_kernel_im2col(ggml_conv_transpose_1d, 1, transpose, _)).
validation_case('conv_transpose_2d',         generate_kernel_im2col(ggml_conv_transpose_2d, 2, transpose, _)).
validation_case('conv_transpose_3d',         generate_kernel_im2col(ggml_conv_transpose_3d, 3, transpose, _)).

%% Family 1: Reductions — 8 ops
validation_case('reduce_sum_rows',  generate_kernel_reduction(ggml_sum_rows, 2, axis_inner, _)).
validation_case('reduce_mean',      generate_kernel_reduction(ggml_mean,     2, axis_inner, _)).
validation_case('reduce_max',       generate_kernel_reduction(ggml_max,      2, axis_inner, _)).
validation_case('reduce_min',       generate_kernel_reduction(ggml_min,      2, axis_inner, _)).
validation_case('reduce_argmax',    generate_kernel_reduction(ggml_argmax,   2, axis_inner, _)).
validation_case('reduce_argmin',    generate_kernel_reduction(ggml_argmin,   2, axis_inner, _)).
validation_case('cumsum',           generate_kernel_reduction(ggml_cumsum,   2, axis_inner, _)).
validation_case('cumprod',          generate_kernel_reduction(ggml_cumprod,  2, axis_inner, _)).

%% Family 2: Norms — 4 × {affine, no-affine} = 8 variants
validation_case('norm_layer_plain', generate_kernel_norm(ggml_norm, 2, false, _)).
validation_case('norm_layer_affine', generate_kernel_norm(ggml_norm, 2, true, _)).
validation_case('norm_rms_plain',   generate_kernel_norm(ggml_rms_norm, 2, false, _)).
validation_case('norm_rms_affine',  generate_kernel_norm(ggml_rms_norm, 2, true, _)).
validation_case('norm_l2_plain',    generate_kernel_norm(ggml_l2_norm, 2, false, _)).
validation_case('norm_l2_affine',   generate_kernel_norm(ggml_l2_norm, 2, true, _)).
validation_case('norm_group_plain', generate_kernel_norm(ggml_group_norm, 2, false, _)).

%% Family 5: Losses — 6 ops
validation_case('loss_mse_mean',         generate_kernel_loss(ggml_mse_loss,           mean, [], _)).
validation_case('loss_mse_sum',          generate_kernel_loss(ggml_mse_loss,           sum,  [], _)).
validation_case('loss_cross_entropy',    generate_kernel_loss(ggml_cross_entropy_loss, mean, [], _)).
validation_case('loss_huber',            generate_kernel_loss(ggml_huber_loss,         mean, [], _)).
validation_case('loss_kl_div',           generate_kernel_loss(ggml_kl_div_loss,        sum,  [], _)).
validation_case('loss_hinge',            generate_kernel_loss(ggml_hinge_loss,         mean, [], _)).
validation_case('loss_triplet_margin',   generate_kernel_loss(ggml_triplet_margin_loss, mean, [], _)).

%% Family 3: Pool — 2D variants (1D/3D are skeletons; only 2D is full body)
validation_case('pool_2d_max',  generate_kernel_pool(ggml_pool_2d, 2, max, [], _)).
validation_case('pool_2d_avg',  generate_kernel_pool(ggml_pool_2d, 2, avg, [], _)).

%% ─────────────────────────────────────────────────────────────────────
%% Validation harness
%% ─────────────────────────────────────────────────────────────────────

run_validation :-
    %% Ensure tmp dir exists
    tmp_dir(TmpDir),
    format(atom(MkdirCmd), 'mkdir -p ~w', [TmpDir]),
    shell(MkdirCmd, _),
    %% Gather all cases
    findall(Name-Goal, validation_case(Name, Goal), Cases),
    length(Cases, NCases),
    format("Validating ~d kernel variants against nvcc -arch=sm_61~n~n", [NCases]),
    %% Run each
    validate_all(Cases, 0, 0, 0, [], Passed, Failed, Skipped, Failures),
    %% Report
    format("~n=============================================~n", []),
    format("nvcc COMPILE VALIDATION RESULTS:~n", []),
    format("  Passed:  ~d~n", [Passed]),
    format("  Failed:  ~d~n", [Failed]),
    format("  Skipped: ~d~n", [Skipped]),
    format("=============================================~n", []),
    ( Failures = []
    -> true
    ;  format("~nFailures detail:~n", []),
       forall(member(F, Failures), format("  ~w~n", [F]))
    ),
    ( Failed > 0 -> halt(1) ; halt(0) ).

validate_all([], P, F, S, Failures, P, F, S, Failures).
validate_all([Name-Goal | Rest], P0, F0, S0, FailAcc, P, F, S, Failures) :-
    %% Try to generate the kernel AST
    catch(
        ( copy_term(Goal, GoalCopy),
          GoalCopy =.. [Functor | Args0],
          %% Replace last unbound with a fresh var to capture K
          last(Args0, _),
          %% Run the goal; the last arg is the kernel AST
          call(GoalCopy)
        ),
        Err,
        ( format("  ERR ~w: generation failed: ~w~n", [Name, Err]),
          P1 = P0, F1 = F0 + 1, S1 = S0,
          NewFails = [Name-(gen_error(Err)) | FailAcc],
          validate_all(Rest, P1, F1, S1, NewFails, P, F, S, Failures), !
        )
    ),
    %% Extract the kernel from the goal's last argument
    GoalCopy =.. [_ | ArgList],
    last(ArgList, KernelAST),
    %% Try to compile
    ( validate_one(Name, KernelAST, Outcome)
    -> ( Outcome = pass
       -> format("  PASS ~w~n", [Name]),
          P1 is P0 + 1, F1 = F0, S1 = S0, NewFails = FailAcc
       ;  Outcome = fail(Reason)
       -> format("  FAIL ~w: ~w~n", [Name, Reason]),
          P1 = P0, F1 is F0 + 1, S1 = S0,
          NewFails = [Name-fail(Reason) | FailAcc]
       ;  Outcome = skip(Reason)
       -> format("  SKIP ~w: ~w~n", [Name, Reason]),
          P1 = P0, F1 = F0, S1 is S0 + 1,
          NewFails = FailAcc
       )
    ;  format("  HARN ~w: harness error~n", [Name]),
       P1 = P0, F1 = F0 + 1, S1 = S0,
       NewFails = [Name-harness_error | FailAcc]
    ),
    validate_all(Rest, P1, F1, S1, NewFails, P, F, S, Failures).

%% validate_one(+Name, +KernelAST, -Outcome)
%%   Outcome: pass | fail(Reason) | skip(Reason)
validate_one(Name, KernelAST, Outcome) :-
    tmp_dir(TmpDir),
    format(atom(CuPath), '~w/~w.cu', [TmpDir, Name]),
    format(atom(ObjPath), '~w/~w.o', [TmpDir, Name]),
    %% Emit
    emit_program([c_include_sys('cuda_runtime.h'), c_blank, KernelAST], Code),
    %% Write
    open(CuPath, write, S),
    write(S, Code),
    close(S),
    %% Compile
    target_arch(Arch),
    cuda_include_flag(IncFlag),
    nvcc_bin(Nvcc),
    format(atom(NvccCmd),
        '~w -arch=~w ~w -c ~w -o ~w 2>&1',
        [Nvcc, Arch, IncFlag, CuPath, ObjPath]),
    %% Capture output
    setup_call_cleanup(
        process_create(path(sh), ['-c', NvccCmd],
                       [stdout(pipe(Out)), stderr(pipe(Err)),
                        process(PID)]),
        ( read_string(Out, _, _StdoutStr),
          read_string(Err, _, _StderrStr),
          process_wait(PID, ExitStatus)
        ),
        ( close(Out), close(Err) )
    ),
    %% Determine outcome
    ( ExitStatus = exit(0)
    -> Outcome = pass
    ;  ExitStatus = exit(N),
       format(atom(Reason), 'nvcc exit ~w', [N]),
       Outcome = fail(Reason)
    ).

%% ─────────────────────────────────────────────────────────────────────
%% Structural-only run (when nvcc isn't available)
%% ─────────────────────────────────────────────────────────────────────
%% Just verifies each kernel generates a valid AST + emit_program succeeds.
%% Doesn't actually compile.

run_structural_only :-
    findall(Name-Goal, validation_case(Name, Goal), Cases),
    length(Cases, NCases),
    format("~nStructural-only run (~d cases): verify AST + emit_program~n~n", [NCases]),
    structural_all(Cases, 0, 0, Passed, Failed),
    format("~n=============================================~n", []),
    format("STRUCTURAL VALIDATION (no nvcc):~n", []),
    format("  Passed: ~d  Failed: ~d~n", [Passed, Failed]),
    format("=============================================~n", []),
    ( Failed > 0 -> halt(1) ; halt(0) ).

structural_all([], P, F, P, F).
structural_all([Name-Goal | Rest], P0, F0, P, F) :-
    ( catch(call(Goal), _, fail)
    -> Goal =.. [_ | Args],
       last(Args, KernelAST),
       ( catch(
            emit_program([c_include_sys('cuda_runtime.h'), c_blank, KernelAST], _),
            _, fail)
       -> format("  PASS ~w (struct)~n", [Name]), P1 is P0 + 1, F1 = F0
       ;  format("  FAIL ~w (emit failed)~n", [Name]), P1 = P0, F1 is F0 + 1
       )
    ; format("  FAIL ~w (gen failed)~n", [Name]), P1 = P0, F1 is F0 + 1
    ),
    structural_all(Rest, P1, F1, P, F).

:- initialization(run_tests, main).
