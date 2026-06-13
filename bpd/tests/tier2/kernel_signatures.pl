%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% Auto-generated kernel signature catalog
%% Subtask 2b — Tier 2 plan 8d65ba1c-5782-47f4-82a3-fa017b727e96
%% Reflectively extracted from substrate's emit predicates.

kernel_signature('conv_2d_forward', 'im2col_2d_forward', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('B', c_scalar(int)),
    ('C', c_scalar(int)),
    ('H', c_scalar(int)),
    ('W', c_scalar(int)),
    (kH, c_scalar(int)),
    (kW, c_scalar(int)),
    (outH, c_scalar(int)),
    (outW, c_scalar(int)),
    (stride_h, c_scalar(int)),
    (stride_w, c_scalar(int)),
    (pad_h, c_scalar(int)),
    (pad_w, c_scalar(int)),
    (dilation_h, c_scalar(int)),
    (dilation_w, c_scalar(int)),
    ]).

kernel_signature('conv_1d_forward', 'im2col_1d_forward', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('B', c_scalar(int)),
    ('C', c_scalar(int)),
    ('L', c_scalar(int)),
    (kL, c_scalar(int)),
    (outL, c_scalar(int)),
    (stride, c_scalar(int)),
    (pad, c_scalar(int)),
    (dilation, c_scalar(int)),
    ]).

kernel_signature('conv_3d_forward', 'im2col_3d_forward', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('B', c_scalar(int)),
    ('C', c_scalar(int)),
    ('D', c_scalar(int)),
    ('H', c_scalar(int)),
    ('W', c_scalar(int)),
    (kD, c_scalar(int)),
    (kH, c_scalar(int)),
    (kW, c_scalar(int)),
    (outD, c_scalar(int)),
    (outH, c_scalar(int)),
    (outW, c_scalar(int)),
    ]).

kernel_signature('conv_transpose_2d', 'col2im_2d_transpose', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ]).

kernel_signature('reduce_sum_rows', 'reduce_sum', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('reduce_mean', 'reduce_mean', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('reduce_max', 'reduce_max', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('reduce_min', 'reduce_min', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('reduce_argmax', 'reduce_argmax', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('reduce_argmin', 'reduce_argmin', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('cumsum', 'cumsum', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('cumprod', 'cumprod', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('norm_layer_plain', 'norm_layer', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ]).

kernel_signature('norm_layer_affine', 'norm_layer', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ('W', c_const_ptr(float)),
    ('B', c_const_ptr(float)),
    ]).

kernel_signature('norm_rms_plain', 'norm_rms', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ]).

kernel_signature('norm_rms_affine', 'norm_rms', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ('W', c_const_ptr(float)),
    ('B', c_const_ptr(float)),
    ]).

kernel_signature('norm_l2_plain', 'norm_l2', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ]).

kernel_signature('norm_l2_affine', 'norm_l2', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ('W', c_const_ptr(float)),
    ('B', c_const_ptr(float)),
    ]).

kernel_signature('norm_group_plain', 'norm_group', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (eps, c_scalar(float)),
    ]).

kernel_signature('loss_mse_mean', 'loss_mse', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('loss_mse_sum', 'loss_mse', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('loss_cross_entropy', 'loss_crossentropy', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('loss_huber', 'loss_huber', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (huber_delta, c_scalar(float)),
    ]).

kernel_signature('loss_kl_div', 'loss_kldiv', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('loss_hinge', 'loss_hinge', [
    ('X', c_const_ptr(float)),
    ('Y', c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    ]).

kernel_signature('loss_triplet_margin', 'loss_triplet_margin', [
    (anchor, c_const_ptr(float)),
    (positive, c_const_ptr(float)),
    (negative, c_const_ptr(float)),
    ('Out', c_ptr(float)),
    ('N', c_scalar(int)),
    (outer, c_scalar(int)),
    (margin, c_scalar(float)),
    ]).

kernel_signature('pool_2d_max', 'pool_2d_max', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('B', c_scalar(int)),
    ('C', c_scalar(int)),
    (inH, c_scalar(int)),
    (inW, c_scalar(int)),
    (outH, c_scalar(int)),
    (outW, c_scalar(int)),
    (kH, c_scalar(int)),
    (kW, c_scalar(int)),
    (stride_h, c_scalar(int)),
    (stride_w, c_scalar(int)),
    (pad_h, c_scalar(int)),
    (pad_w, c_scalar(int)),
    ]).

kernel_signature('pool_2d_avg', 'pool_2d_avg', [
    ('X', c_const_ptr(float)),
    ('Y', c_ptr(float)),
    ('B', c_scalar(int)),
    ('C', c_scalar(int)),
    (inH, c_scalar(int)),
    (inW, c_scalar(int)),
    (outH, c_scalar(int)),
    (outW, c_scalar(int)),
    (kH, c_scalar(int)),
    (kW, c_scalar(int)),
    (stride_h, c_scalar(int)),
    (stride_w, c_scalar(int)),
    (pad_h, c_scalar(int)),
    (pad_w, c_scalar(int)),
    ]).

