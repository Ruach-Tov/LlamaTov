%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% kernelbench_l1_problems.pl — Stanford KernelBench L1 problem definitions.
%% 100 single-operator problems. Per medayek's L3 analysis:
%% "L1 validates op classification completeness, not fusion."
%%
%% Source: github.com/ScalingIntelligence/KernelBench/tree/main/KernelBench/level1
%% Authored: metayen 2026-05-15
%% Format matches kernelbench_l2_problems.pl (mavchin) for tooling consistency.
%%
%% Each problem is a single op. Op kinds map to the substrate's ggml taxonomy
%% (the same one used by fusion_to_cuda.pl).
%%
%% Split convention follows L2: odd=TRAIN, even=TEST.
%%
%% Op-kind mappings (substrate-honest decisions):
%%   Matmul variants (1-18)    → ggml_mul_mat (the L1 matmul kernel covers
%%                                all variants; dimension differences are
%%                                handled by the same kernel template)
%%   Activations (19-32)       → specific ggml op per activation
%%   Norms (33-40)             → ggml_norm (substrate currently classifies
%%                                all norms uniformly; dimension/group config
%%                                handled per-problem at codegen time)
%%   Pooling (41-46)           → ggml_pool (1D/2D/3D variants share family)
%%   Reductions (47-53)        → ggml_sum_rows / ggml_mean / ggml_argmax
%%   Convolutions (50, 54-87)  → ggml_conv (covers Conv2D, Conv3D, ConvTranspose,
%%                                Conv1D, ConvDepthwise — substrate distinguishes
%%                                via op_params not via op_kind)
%%   Cumulative (89-93)        → ggml_cumsum
%%   Losses (94-100)           → loss-specific kinds where possible

:- module(kernelbench_l1, [kb_problem/4]).

%% kb_problem(+Number, +Split, +Name, +Ops)
%%   Ops is a list of op(name, ggml_kind, sequence_number) terms.
%%   For L1, this list always has exactly one element.

%% ─────────────────────────────────────────────────────────────────────
%% Matrix multiplication variants (1-18)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(1,  train, '1_Square_matrix_multiplication',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(2,  test,  '2_Standard_matrix_multiplication',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(3,  train, '3_Batched_matrix_multiplication',
    [op(bmm_1, ggml_mul_mat, 1)]).
kb_problem(4,  test,  '4_Matrix_vector_multiplication',
    [op(matvec_1, ggml_mul_mat, 1)]).
kb_problem(5,  train, '5_Matrix_scalar_multiplication',
    [op(matscale_1, ggml_scale, 1)]).
kb_problem(6,  test,  '6_Matmul_with_large_K_dimension',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(7,  train, '7_Matmul_with_small_K_dimension',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(8,  test,  '8_Matmul_with_irregular_shapes',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(9,  train, '9_Tall_skinny_matrix_multiplication',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(10, test,  '10_3D_tensor_matrix_multiplication',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(11, train, '11_4D_tensor_matrix_multiplication',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(12, test,  '12_Matmul_with_diagonal_matrices',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(13, train, '13_Matmul_for_symmetric_matrices',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(14, test,  '14_Matmul_for_upper_triangular_matrices',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(15, train, '15_Matmul_for_lower_triangular_matrices',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(16, test,  '16_Matmul_with_transposed_A',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(17, train, '17_Matmul_with_transposed_B',
    [op(matmul_1, ggml_mul_mat, 1)]).
kb_problem(18, test,  '18_Matmul_with_transposed_both',
    [op(matmul_1, ggml_mul_mat, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Activations (19-32)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(19, train, '19_ReLU',
    [op(relu_1, ggml_relu, 1)]).
kb_problem(20, test,  '20_LeakyReLU',
    [op(leakyrelu_1, ggml_leaky_relu, 1)]).
kb_problem(21, train, '21_Sigmoid',
    [op(sigmoid_1, ggml_sigmoid, 1)]).
kb_problem(22, test,  '22_Tanh',
    [op(tanh_1, ggml_tanh, 1)]).
kb_problem(23, train, '23_Softmax',
    [op(softmax_1, ggml_soft_max_ext, 1)]).
kb_problem(24, test,  '24_LogSoftmax',
    [op(logsoftmax_1, ggml_log_soft_max, 1)]).
kb_problem(25, train, '25_Swish',
    [op(swish_1, ggml_silu, 1)]).                   % Swish = SiLU
kb_problem(26, test,  '26_GELU',
    [op(gelu_1, ggml_gelu, 1)]).
kb_problem(27, train, '27_SELU',
    [op(selu_1, ggml_selu, 1)]).
kb_problem(28, test,  '28_HardSigmoid',
    [op(hardsigmoid_1, ggml_hardsigmoid, 1)]).
kb_problem(29, train, '29_Softplus',
    [op(softplus_1, ggml_softplus, 1)]).
kb_problem(30, test,  '30_Softsign',
    [op(softsign_1, ggml_softsign, 1)]).
kb_problem(31, train, '31_ELU',
    [op(elu_1, ggml_elu, 1)]).
kb_problem(32, test,  '32_HardTanh',
    [op(hardtanh_1, ggml_clamp, 1)]).               % HardTanh = clamp(-1, 1)

%% ─────────────────────────────────────────────────────────────────────
%% Normalizations (33-40)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(33, train, '33_BatchNorm',
    [op(batchnorm_1, ggml_norm, 1)]).
kb_problem(34, test,  '34_InstanceNorm',
    [op(instancenorm_1, ggml_norm, 1)]).
kb_problem(35, train, '35_GroupNorm',
    [op(groupnorm_1, ggml_group_norm, 1)]).
kb_problem(36, test,  '36_RMSNorm',
    [op(rmsnorm_1, ggml_rms_norm, 1)]).
kb_problem(37, train, '37_FrobeniusNorm',
    [op(frobeniusnorm_1, ggml_norm, 1)]).
kb_problem(38, test,  '38_L1Norm',
    [op(l1norm_1, ggml_norm, 1)]).
kb_problem(39, train, '39_L2Norm',
    [op(l2norm_1, ggml_l2_norm, 1)]).
kb_problem(40, test,  '40_LayerNorm',
    [op(layernorm_1, ggml_norm, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Pooling (41-46)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(41, train, '41_Max_Pooling_1D',
    [op(maxpool1d_1, ggml_pool_1d, 1)]).
kb_problem(42, test,  '42_Max_Pooling_2D',
    [op(maxpool2d_1, ggml_pool_2d, 1)]).
kb_problem(43, train, '43_Max_Pooling_3D',
    [op(maxpool3d_1, ggml_pool_3d, 1)]).
kb_problem(44, test,  '44_Average_Pooling_1D',
    [op(avgpool1d_1, ggml_pool_1d, 1)]).
kb_problem(45, train, '45_Average_Pooling_2D',
    [op(avgpool2d_1, ggml_pool_2d, 1)]).
kb_problem(46, test,  '46_Average_Pooling_3D',
    [op(avgpool3d_1, ggml_pool_3d, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Reductions (47-49, 51-53)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(47, train, '47_Sum_reduction_over_a_dimension',
    [op(sum_1, ggml_sum_rows, 1)]).
kb_problem(48, test,  '48_Mean_reduction_over_a_dimension',
    [op(mean_1, ggml_mean, 1)]).
kb_problem(49, train, '49_Max_reduction_over_a_dimension',
    [op(max_1, ggml_max, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Convolution: 2D standard (50)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(50, test,  '50_conv_standard_2D_square_input_square_kernel',
    [op(conv2d_1, ggml_conv_2d, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% More reductions (51-53)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(51, train, '51_Argmax_over_a_dimension',
    [op(argmax_1, ggml_argmax, 1)]).
kb_problem(52, test,  '52_Argmin_over_a_dimension',
    [op(argmin_1, ggml_argmin, 1)]).
kb_problem(53, train, '53_Min_reduction_over_a_dimension',
    [op(min_1, ggml_min, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Convolutions: 3D standard, 2D variants, 1D, transposed (54-66)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(54, test,  '54_conv_standard_3D_square_input_square_kernel',
    [op(conv3d_1, ggml_conv_3d, 1)]).
kb_problem(55, train, '55_conv_standard_2D_asymmetric_input_square_kernel',
    [op(conv2d_1, ggml_conv_2d, 1)]).
kb_problem(56, test,  '56_conv_standard_2D_asymmetric_input_asymmetric_kernel',
    [op(conv2d_1, ggml_conv_2d, 1)]).
kb_problem(57, train, '57_conv_transposed_2D_square_input_square_kernel',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(58, test,  '58_conv_transposed_3D_asymmetric_input_asymmetric_kernel',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(59, train, '59_conv_standard_3D_asymmetric_input_square_kernel',
    [op(conv3d_1, ggml_conv_3d, 1)]).
kb_problem(60, test,  '60_conv_standard_3D_square_input_asymmetric_kernel',
    [op(conv3d_1, ggml_conv_3d, 1)]).
kb_problem(61, train, '61_conv_transposed_3D_square_input_square_kernel',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(62, test,  '62_conv_standard_2D_square_input_asymmetric_kernel',
    [op(conv2d_1, ggml_conv_2d, 1)]).
kb_problem(63, train, '63_conv_standard_2D_square_input_square_kernel',
    [op(conv2d_1, ggml_conv_2d, 1)]).
kb_problem(64, test,  '64_conv_transposed_1D',
    [op(convtranspose1d_1, ggml_conv_transpose_1d, 1)]).
kb_problem(65, train, '65_conv_transposed_2D_square_input_asymmetric_kernel',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(66, test,  '66_conv_standard_3D_asymmetric_input_asymmetric_kernel',
    [op(conv3d_1, ggml_conv_3d, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Convolutions: 1D, more transposed variants (67-81)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(67, train, '67_conv_standard_1D',
    [op(conv1d_1, ggml_conv_1d, 1)]).
kb_problem(68, test,  '68_conv_transposed_3D_square_input_asymmetric_kernel',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(69, train, '69_conv_transposed_2D_asymmetric_input_asymmetric_kernel',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(70, test,  '70_conv_transposed_3D_asymmetric_input_square_kernel',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(71, train, '71_conv_transposed_2D_asymmetric_input_square_kernel',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(72, test,
    '72_conv_transposed_3D_asymmetric_input_asymmetric_kernel_strided_padded_grouped',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(73, train,
    '73_conv_transposed_3D_asymmetric_input_square_kernel_strided_padded_grouped',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(74, test,  '74_conv_transposed_1D_dilated',
    [op(convtranspose1d_1, ggml_conv_transpose_1d, 1)]).
kb_problem(75, train,
    '75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided_grouped_padded_dilated',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(76, test,  '76_conv_standard_1D_dilated_strided',
    [op(conv1d_1, ggml_conv_1d, 1)]).
kb_problem(77, train,
    '77_conv_transposed_3D_square_input_square_kernel_padded_dilated_strided',
    [op(convtranspose3d_1, ggml_conv_transpose_3d, 1)]).
kb_problem(78, test,
    '78_conv_transposed_2D_asymmetric_input_asymmetric_kernel_padded',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).
kb_problem(79, train,
    '79_conv_transposed_1D_asymmetric_input_square_kernel_padded_strided_dilated',
    [op(convtranspose1d_1, ggml_conv_transpose_1d, 1)]).
kb_problem(80, test,
    '80_conv_standard_2D_square_input_asymmetric_kernel_dilated_padded',
    [op(conv2d_1, ggml_conv_2d, 1)]).
kb_problem(81, train,
    '81_conv_transposed_2D_asymmetric_input_square_kernel_dilated_padded_strided',
    [op(convtranspose2d_1, ggml_conv_transpose_2d, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Depthwise / pointwise convolutions (82-87)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(82, test,  '82_conv_depthwise_2D_square_input_square_kernel',
    [op(conv2d_depthwise_1, ggml_conv_2d, 1)]).
kb_problem(83, train, '83_conv_depthwise_2D_square_input_asymmetric_kernel',
    [op(conv2d_depthwise_1, ggml_conv_2d, 1)]).
kb_problem(84, test,  '84_conv_depthwise_2D_asymmetric_input_square_kernel',
    [op(conv2d_depthwise_1, ggml_conv_2d, 1)]).
kb_problem(85, train, '85_conv_depthwise_2D_asymmetric_input_asymmetric_kernel',
    [op(conv2d_depthwise_1, ggml_conv_2d, 1)]).
kb_problem(86, test,  '86_conv_depthwise_separable_2D',
    [op(conv2d_depthwise_separable_1, ggml_conv_2d, 1)]).
kb_problem(87, train, '87_conv_pointwise_2D',
    [op(conv2d_pointwise_1, ggml_conv_2d, 1)]).

%% ─────────────────────────────────────────────────────────────────────
%% Special activations + cumulative + losses (88-100)
%% ─────────────────────────────────────────────────────────────────────

kb_problem(88, test,  '88_MinGPTNewGelu',
    [op(gelu_1, ggml_gelu, 1)]).
kb_problem(89, train, '89_cumsum',
    [op(cumsum_1, ggml_cumsum, 1)]).
kb_problem(90, test,  '90_cumprod',
    [op(cumprod_1, ggml_cumprod, 1)]).
kb_problem(91, train, '91_cumsum_reverse',
    [op(cumsum_reverse_1, ggml_cumsum, 1)]).
kb_problem(92, test,  '92_cumsum_exclusive',
    [op(cumsum_exclusive_1, ggml_cumsum, 1)]).
kb_problem(93, train, '93_masked_cumsum',
    [op(masked_cumsum_1, ggml_cumsum, 1)]).
kb_problem(94, test,  '94_MSELoss',
    [op(mseloss_1, ggml_mse_loss, 1)]).
kb_problem(95, train, '95_CrossEntropyLoss',
    [op(crossentropy_1, ggml_cross_entropy_loss, 1)]).
kb_problem(96, test,  '96_HuberLoss',
    [op(huberloss_1, ggml_huber_loss, 1)]).
kb_problem(97, train, '97_ScaledDotProductAttention',
    [op(sdpa_1, ggml_flash_attn_ext, 1)]).
kb_problem(98, test,  '98_KLDivLoss',
    [op(kldiv_1, ggml_kl_div_loss, 1)]).
kb_problem(99, train, '99_TripletMarginLoss',
    [op(tripletmargin_1, ggml_triplet_margin_loss, 1)]).
kb_problem(100, test, '100_HingeLoss',
    [op(hingeloss_1, ggml_hinge_loss, 1)]).
