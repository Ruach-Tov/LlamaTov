%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
:- use_module("lib/derive_epilogues").

%% Generate fused epilogue chains for all KernelBench L2 patterns
%% that have elementwise sequences after a spatial op.

main :-
    format("=== Auto-Generated L2 Fusion Epilogues ===~n~n"),
    
    Chains = [
        (1,  "Conv2D+ReLU+BiasAdd",           [relu]),
        (4,  "Conv2d+Mish+Mish",               [mish, mish]),
        (5,  "ConvT2d+Sub+Tanh",               [neg, tanh]),
        (7,  "Conv3d+ReLU+LeakyReLU+GELU+Sigmoid", [relu, leaky_relu, gelu, sigmoid]),
        (9,  "Matmul+Sub+Mul+ReLU",            [neg, relu]),
        (12, "Gemm+Mul+LeakyReLU",             [leaky_relu]),
        (16, "ConvT2d+Mish+Hardtanh",          [mish, hardtanh]),
        (26, "ConvT3d+HardSwish",              [hardswish]),
        (29, "Matmul+Mish+Mish",               [mish, mish]),
        (35, "Conv2d+Sub+HardSwish+Mish",      [neg, hardswish, mish]),
        (47, "Conv3d+Mish+Tanh",               [mish, tanh]),
        (48, "Conv3d+Tanh+Sigmoid",            [tanh, sigmoid]),
        (53, "Gemm+Hardtanh+GELU",             [hardtanh, gelu]),
        (54, "Conv2d+LeakyReLU+GELU",          [leaky_relu, gelu]),
        (57, "Conv2d+ReLU+HardSwish",          [relu, hardswish]),
        (59, "Matmul+SiLU",                    [silu]),
        (63, "Gemm+ReLU",                      [relu]),
        (69, "Conv2d+HardSwish+ReLU",          [hardswish, relu]),
        (71, "Conv2d+LeakyReLU",               [leaky_relu]),
        (76, "Gemm+ReLU",                      [relu]),
        (81, "Gemm+SiLU+Tanh",                [silu, tanh]),
        (86, "Matmul+GELU",                    [gelu]),
        (87, "Conv2d+Mish",                    [mish]),
        (90, "Conv3d+LeakyReLU+GELU",          [leaky_relu, gelu]),
        (95, "Matmul+SiLU+Tanh+GELU+Hardtanh", [silu, tanh, gelu, hardtanh])
    ],
    
    %% Generate C file
    open('/tmp/l2_fused_epilogues.c', write, S),
    format(S, "#include <math.h>~n~n", []),
    
    N_generated = 0,
    forall(
        member((Num, Label, Ops), Chains),
        (   derive_chain(Ops, c_var(x), CAST),
            format(S, "/* L2 #~w: ~w */~n", [Num, Label]),
            atom_concat('l2_', Num, FnBase),
            atom_concat(FnBase, '_epilogue', FnName),
            format(S, "static inline float ~w(float x) { return ", [FnName]),
            emit_c(S, CAST),
            format(S, "; }~n~n", []),
            format("  #~w ~w: ", [Num, Label]),
            emit_c(user_output, CAST),
            nl
        )
    ),
    
    close(S),
    nl,
    format("Generated /tmp/l2_fused_epilogues.c~n"),
    format("25 fused epilogue functions ready for compilation~n").

:- main, halt.
