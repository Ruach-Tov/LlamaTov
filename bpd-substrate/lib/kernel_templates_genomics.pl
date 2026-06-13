%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% ═══════════════════════════════════════════════════════════════════════
%% kernel_templates_genomics.pl — Genomics kernels for BPD substrate
%%
%% Smith-Waterman local sequence alignment expressed as Prolog facts.
%% The substrate's first application outside AI inference.
%%
%% Licensed under GPLv2 (not RTAAL — this is substrate, not optimizer)
%% ═══════════════════════════════════════════════════════════════════════

/** <module> Genomics Kernel Templates

Smith-Waterman local sequence alignment expressed as declarative Prolog
facts. Each fact describes the mathematical operation; the code generator
emits C or CUDA from these facts.

The Smith-Waterman algorithm is a tiled 2D dynamic programming kernel:
  H(i,j) = max(0, H(i-1,j-1) + score(q[i], r[j]), E(i,j), F(i,j))
  E(i,j) = max(H(i,j-1) - gap_open, E(i,j-1) - gap_extend)
  F(i,j) = max(H(i-1,j) - gap_open, F(i-1,j) - gap_extend)

This decomposition mirrors how we express neural network ops:
  - Each matrix update is a named operation
  - Dependencies between cells define the compute graph
  - Anti-diagonal parallelism is derivable from the dependencies
  - Tile size is a sweepable substrate-design parameter

@author Ruach Tov Collective
@see bench/bpd_smith_waterman.c for the CPU reference implementation
@see bench/verify_smith_waterman.py for 145-test verification harness
*/

:- module(kernel_templates_genomics, [
    sw_scoring_function/3,      % +Name, +Params, -ScoreExpr
    sw_cell_update/2,           % +Name, -CellFacts
    sw_gap_model/3,             % +Name, +GapType, -GapExpr
    sw_parallelism/2,           % +Name, -ParallelismFacts
    sw_kernel_params/2,         % +Name, -DefaultParams
    sw_traceback_op/2,          % +Name, -TracebackFacts
    sa_construction_op/2,       % +Name, -ConstructionFacts
    lcp_construction_op/2,      % +Name, -LcpFacts
    repeat_enumeration_op/3,    % +Name, +Params, -EnumFacts
    pwm_score_op/3,             % +Name, +Params, -ScoreFacts
    pwm_build_op/3,             % +Name, +Params, -BuildFacts
    motif_sample_op/2,          % +Name, -SampleFacts
    convergence_op/2            % +Name, -ConvFacts
]).

%% ═══════════════════════════════════════════════════════════════════════
%% Scoring function: how to compare two sequence elements
%% ═══════════════════════════════════════════════════════════════════════

%! sw_scoring_function(+Name, +Params, -ScoreExpr) is det.
%  Define the scoring function for a Smith-Waterman variant.
%  ScoreExpr is a c_ast term that computes the substitution score.

%% DNA scoring: match/mismatch (simplest case)
sw_scoring_function(dna_simple,
    [param(c_type(char), qi), param(c_type(char), rj),
     param(c_type(int), match), param(c_type(int), mismatch)],
    c_ternary(
        c_binop('==', c_var(qi), c_var(rj)),
        c_var(match),
        c_var(mismatch))).

%% DNA scoring via lookup table (4x4 substitution matrix)
sw_scoring_function(dna_matrix,
    [param(c_type(char), qi), param(c_type(char), rj),
     param(c_type(const_ptr(c_type(int))), subst_matrix)],
    c_index(c_var(subst_matrix),
        c_binop('+',
            c_binop('*', c_call(base_to_idx, [c_var(qi)]), c_int(4)),
            c_call(base_to_idx, [c_var(rj)])))).

%% Protein scoring via BLOSUM62 (20x20 substitution matrix)
sw_scoring_function(protein_blosum62,
    [param(c_type(char), qi), param(c_type(char), rj),
     param(c_type(const_ptr(c_type(int))), blosum62)],
    c_index(c_var(blosum62),
        c_binop('+',
            c_binop('*', c_call(aa_to_idx, [c_var(qi)]), c_int(20)),
            c_call(aa_to_idx, [c_var(rj)])))).

%% ═══════════════════════════════════════════════════════════════════════
%% Cell update: the core recurrence relation
%% ═══════════════════════════════════════════════════════════════════════

%! sw_cell_update(+Name, -CellFacts) is det.
%  Define the cell update rule for Smith-Waterman.
%  CellFacts describes the dependencies and computation for H(i,j).

sw_cell_update(affine_gaps, cell_facts(
    %% Dependencies: which cells are read to compute H(i,j)
    dependencies([
        dep(h_diag, c_index_2d(h_matrix, c_binop('-', c_var(i), c_int(1)),
                                          c_binop('-', c_var(j), c_int(1)))),
        dep(h_left, c_index_2d(h_matrix, c_var(i),
                                          c_binop('-', c_var(j), c_int(1)))),
        dep(h_up,   c_index_2d(h_matrix, c_binop('-', c_var(i), c_int(1)),
                                          c_var(j)))
    ]),

    %% E update: horizontal gap (gap in query)
    e_update(c_call(max, [
        c_binop('-', c_var(h_left), c_var(gap_open)),
        c_binop('-', c_var(e_left), c_var(gap_extend))
    ])),

    %% F update: vertical gap (gap in reference)
    f_update(c_call(max, [
        c_binop('-', c_var(h_up), c_var(gap_open)),
        c_binop('-', c_var(f_up), c_var(gap_extend))
    ])),

    %% H update: the max of all options
    h_update(c_call(max, [
        c_int(0),
        c_binop('+', c_var(h_diag), c_var(score)),
        c_var(e_val),
        c_var(f_val)
    ]))
)).

%% Linear gaps (simpler, no E/F matrices)
sw_cell_update(linear_gaps, cell_facts(
    dependencies([
        dep(h_diag, c_index_2d(h_matrix, c_binop('-', c_var(i), c_int(1)),
                                          c_binop('-', c_var(j), c_int(1)))),
        dep(h_left, c_index_2d(h_matrix, c_var(i),
                                          c_binop('-', c_var(j), c_int(1)))),
        dep(h_up,   c_index_2d(h_matrix, c_binop('-', c_var(i), c_int(1)),
                                          c_var(j)))
    ]),
    e_update(none),
    f_update(none),
    h_update(c_call(max, [
        c_int(0),
        c_binop('+', c_var(h_diag), c_var(score)),
        c_binop('-', c_var(h_left), c_var(gap_penalty)),
        c_binop('-', c_var(h_up), c_var(gap_penalty))
    ]))
)).

%% ═══════════════════════════════════════════════════════════════════════
%% Gap model: how gaps are penalized
%% ═══════════════════════════════════════════════════════════════════════

%! sw_gap_model(+Name, +GapType, -GapExpr) is det.
%  Define gap penalty model.

sw_gap_model(affine, open_cost, c_var(gap_open)).
sw_gap_model(affine, extend_cost, c_var(gap_extend)).
sw_gap_model(linear, open_cost, c_var(gap_penalty)).
sw_gap_model(linear, extend_cost, c_var(gap_penalty)).

%% ═══════════════════════════════════════════════════════════════════════
%% Parallelism: how the DP matrix can be computed in parallel
%% ═══════════════════════════════════════════════════════════════════════

%! sw_parallelism(+Name, -ParallelismFacts) is det.
%  Describe the parallelism available in the algorithm.

sw_parallelism(anti_diagonal, parallelism_facts(
    %% Cells on the same anti-diagonal (i+j = const) are independent
    parallel_dimension(anti_diagonal),
    %% Anti-diagonal d has min(d, qlen, rlen, qlen+rlen-d) cells
    parallel_width_expr(c_call(min, [
        c_var(d),
        c_var(qlen),
        c_var(rlen),
        c_binop('-', c_binop('+', c_var(qlen), c_var(rlen)), c_var(d))
    ])),
    %% Total anti-diagonals = qlen + rlen - 1
    total_steps(c_binop('-', c_binop('+', c_var(qlen), c_var(rlen)), c_int(1))),
    %% GPU mapping: one thread per cell on the anti-diagonal
    gpu_mapping(one_thread_per_cell)
)).

%% ═══════════════════════════════════════════════════════════════════════
%% Default parameters: sweepable substrate-design parameters
%% ═══════════════════════════════════════════════════════════════════════

%! sw_kernel_params(+Name, -DefaultParams) is det.
%  Default kernel parameters. These are sweepable.

sw_kernel_params(gpu_default, [
    tile_width(32),         %% cells per thread block on anti-diagonal
    block_size(128),        %% threads per block
    shared_mem_rows(2),     %% rows of H matrix in shared memory
    match_score(2),
    mismatch_score(-1),
    gap_open(3),
    gap_extend(1)
]).

sw_kernel_params(cpu_default, [
    tile_width(64),         %% SIMD width for striped SW
    simd_lanes(4),          %% SSE = 4 floats, AVX2 = 8
    match_score(2),
    mismatch_score(-1),
    gap_open(3),
    gap_extend(1)
]).

%% ═══════════════════════════════════════════════════════════════════════
%% Traceback: how to reconstruct the alignment from the score matrix
%% ═══════════════════════════════════════════════════════════════════════

%! sw_traceback_op(+Name, -TracebackFacts) is det.
%  Describe the traceback operation.

sw_traceback_op(standard, traceback_facts(
    %% Start from the cell with maximum H score
    start(max_cell),
    %% At each step, follow the predecessor
    step_rule([
        if(c_binop('==', c_var(h_ij),
            c_binop('+', c_var(h_diag), c_var(score))),
            action(match, move(-1, -1))),
        if(c_binop('==', c_var(h_ij), c_var(e_ij)),
            action(deletion, move(0, -1))),
        if(c_binop('==', c_var(h_ij), c_var(f_ij)),
            action(insertion, move(-1, 0)))
    ]),
    %% Stop when H reaches 0
    stop_condition(c_binop('<=', c_var(h_ij), c_int(0))),
    %% Output format: CIGAR string
    output_format(cigar)
)).

%% ═══════════════════════════════════════════════════════════════════════
%% Suffix array / LCP construction — string indexing primitives
%% ═══════════════════════════════════════════════════════════════════════

%! sa_construction_op(+Name, -ConstructionFacts) is det.
%  Describe suffix array construction algorithm and parallelism.

sa_construction_op(gpu_radix, construction_facts(
    algorithm(radix_sort_32bit),
    key_encoding(first_4_bytes_packed_uint32),
    parallelism(global_three_phase_scan),
    sort_passes(32),
    work_complexity('O(n * 32)')
)).

sa_construction_op(cpu_naive, construction_facts(
    algorithm(qsort_suffix_compare),
    parallelism(sequential),
    work_complexity('O(n * log(n) * n)')
)).

%! lcp_construction_op(+Name, -LcpFacts) is det.
%  Describe LCP array construction.

lcp_construction_op(cpu_kasai, lcp_facts(
    algorithm(kasai_linear),
    dependencies(sequential_via_rank_array),
    work_complexity('O(n)')
)).

lcp_construction_op(gpu_phi, lcp_facts(
    algorithm(phi_array_parallel),
    dependencies(embarrassingly_parallel_per_position),
    steps([
        build_phi,           %% phi[SA[i]] = SA[i-1], parallel
        compute_plcp,        %% compare text[i..] vs text[phi[i]..], parallel
        permute_to_sa_order  %% LCP[i] = plcp[SA[i]], parallel
    ]),
    work_complexity('O(n * avg_lcp)')
)).

%! repeat_enumeration_op(+Name, +Params, -EnumFacts) is det.
%  Describe how to find repeated substrings from SA + LCP.

repeat_enumeration_op(lcp_intervals, [min_len(MinLen), min_count(MinCount)],
    enum_facts(
        algorithm(lcp_interval_scan),
        input(suffix_array, lcp_array),
        filter(lcp_value >= MinLen, interval_width >= MinCount),
        output(candidate_list(sa_start, sa_end, repeat_length, count))
    )).

%% ═══════════════════════════════════════════════════════════════════════
%% Position Weight Matrix — motif scoring primitives
%% ═══════════════════════════════════════════════════════════════════════

%! pwm_score_op(+Name, +Params, -ScoreFacts) is det.
%  Describe PWM scoring of all positions in a sequence.

pwm_score_op(log_odds,
    [param(c_type(int), motif_width), param(c_type(int), alphabet_size)],
    score_facts(
        per_position_score(
            c_sum(j, 0, c_var(motif_width),
                c_index(c_var(pwm),
                    c_binop('+',
                        c_binop('*', c_var(j), c_var(alphabet_size)),
                        c_call(base_idx, [c_index(c_var(seq), c_binop('+', c_var(pos), c_var(j)))]))))),
        parallelism(one_thread_per_position),
        gpu_mapping(grid_x_is_positions, grid_y_is_sequences)
    )).

%! pwm_build_op(+Name, +Params, -BuildFacts) is det.
%  Describe PWM construction from aligned sequences.

pwm_build_op(count_to_log_odds,
    [param(c_type(float), pseudocount)],
    build_facts(
        step1(count_bases_per_position),
        step2(add_pseudocount),
        step3(normalize_to_frequency),
        step4(log_odds_vs_background),
        formula(c_call(logf, [c_binop('/',
            c_binop('/', c_binop('+', c_var(count), c_var(pseudocount)), c_var(total)),
            c_var(background))]))
    )).

%! motif_sample_op(+Name, -SampleFacts) is det.
%  Describe position sampling for Gibbs iteration.

motif_sample_op(gibbs_collapsed,
    sample_facts(
        strategy(leave_one_out),
        dp_insight(subtract_held_out_from_total_counts),
        score_transform(exp_minus_max),
        sampling(cumulative_sum_binary_search),
        parallelism(all_sequences_simultaneously),
        gpu_benefit('O(1) in number of sequences due to DP scheduling')
    )).

%! convergence_op(+Name, -ConvFacts) is det.
%  Describe convergence detection for iterative motif discovery.

convergence_op(frobenius_norm,
    convergence_facts(
        metric(frobenius_norm_of_pwm_delta),
        formula(c_call(sqrtf, [c_sum(j, 0, c_var(motif_width),
            c_sum(b, 0, 4,
                c_binop('*',
                    c_binop('-', c_index_2d(pwm_new, j, b), c_index_2d(pwm_old, j, b)),
                    c_binop('-', c_index_2d(pwm_new, j, b), c_index_2d(pwm_old, j, b)))))])),
        threshold(0.001),
        check_interval(every_iteration)
    )).
