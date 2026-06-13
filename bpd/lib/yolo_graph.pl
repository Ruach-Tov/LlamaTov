%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% warning-free consult (canary audibility):
:- discontiguous op_attr/3, op_inputs/2, op_output/2.
%% YOLOv5n compute graph - auto-generated from trained model
:- discontiguous op_kind/2, op_output/2, op_inputs/2, op_attr/3.
% 199 ops total

op_kind(l0_conv, conv2d).
op_output(l0_conv, l0_conv_out).
op_inputs(l0_conv, [input]).
op_attr(l0_conv, kernel_size, 6).
op_attr(l0_conv, in_channels, 3).
op_attr(l0_conv, out_channels, 16).
op_kind(l0_bn, batchnorm).
op_output(l0_bn, l0_bn_out).
op_inputs(l0_bn, [l0_conv_out]).
op_kind(l0_silu, silu).
op_output(l0_silu, l0_silu_out).
op_inputs(l0_silu, [l0_bn_out]).
op_kind(l1_conv, conv2d).
op_output(l1_conv, l1_conv_out).
op_inputs(l1_conv, [l0_silu_out]).
op_attr(l1_conv, kernel_size, 3).
op_attr(l1_conv, in_channels, 16).
op_attr(l1_conv, out_channels, 32).
op_kind(l1_bn, batchnorm).
op_output(l1_bn, l1_bn_out).
op_inputs(l1_bn, [l1_conv_out]).
op_kind(l1_silu, silu).
op_output(l1_silu, l1_silu_out).
op_inputs(l1_silu, [l1_bn_out]).
op_kind(l2_cv1_conv, conv2d).
op_output(l2_cv1_conv, l2_cv1_conv_out).
op_inputs(l2_cv1_conv, [l1_silu_out]).
op_attr(l2_cv1_conv, kernel_size, 1).
op_attr(l2_cv1_conv, in_channels, 32).
op_attr(l2_cv1_conv, out_channels, 16).
op_kind(l2_cv1_bn, batchnorm).
op_output(l2_cv1_bn, l2_cv1_bn_out).
op_inputs(l2_cv1_bn, [l2_cv1_conv_out]).
op_kind(l2_cv1_silu, silu).
op_output(l2_cv1_silu, l2_cv1_silu_out).
op_inputs(l2_cv1_silu, [l2_cv1_bn_out]).
op_kind(l2_cv2_conv, conv2d).
op_output(l2_cv2_conv, l2_cv2_conv_out).
op_inputs(l2_cv2_conv, [l1_silu_out]).
op_attr(l2_cv2_conv, kernel_size, 1).
op_attr(l2_cv2_conv, in_channels, 32).
op_attr(l2_cv2_conv, out_channels, 16).
op_kind(l2_cv2_bn, batchnorm).
op_output(l2_cv2_bn, l2_cv2_bn_out).
op_inputs(l2_cv2_bn, [l2_cv2_conv_out]).
op_kind(l2_cv2_silu, silu).
op_output(l2_cv2_silu, l2_cv2_silu_out).
op_inputs(l2_cv2_silu, [l2_cv2_bn_out]).
op_kind(l2_bot0_cv1_conv, conv2d).
op_output(l2_bot0_cv1_conv, l2_bot0_cv1_conv_out).
op_inputs(l2_bot0_cv1_conv, [l2_cv1_silu_out]).
op_attr(l2_bot0_cv1_conv, kernel_size, 1).
op_attr(l2_bot0_cv1_conv, in_channels, 16).
op_attr(l2_bot0_cv1_conv, out_channels, 16).
op_kind(l2_bot0_cv1_bn, batchnorm).
op_output(l2_bot0_cv1_bn, l2_bot0_cv1_bn_out).
op_inputs(l2_bot0_cv1_bn, [l2_bot0_cv1_conv_out]).
op_kind(l2_bot0_cv1_silu, silu).
op_output(l2_bot0_cv1_silu, l2_bot0_cv1_silu_out).
op_inputs(l2_bot0_cv1_silu, [l2_bot0_cv1_bn_out]).
op_kind(l2_bot0_cv2_conv, conv2d).
op_output(l2_bot0_cv2_conv, l2_bot0_cv2_conv_out).
op_inputs(l2_bot0_cv2_conv, [l2_bot0_cv1_silu_out]).
op_attr(l2_bot0_cv2_conv, kernel_size, 3).
op_attr(l2_bot0_cv2_conv, in_channels, 16).
op_attr(l2_bot0_cv2_conv, out_channels, 16).
op_kind(l2_bot0_cv2_bn, batchnorm).
op_output(l2_bot0_cv2_bn, l2_bot0_cv2_bn_out).
op_inputs(l2_bot0_cv2_bn, [l2_bot0_cv2_conv_out]).
op_kind(l2_bot0_cv2_silu, silu).
op_output(l2_bot0_cv2_silu, l2_bot0_cv2_silu_out).
op_inputs(l2_bot0_cv2_silu, [l2_bot0_cv2_bn_out]).
op_kind(l2_bot0_add, add).
op_output(l2_bot0_add, l2_bot0_add_out).
op_inputs(l2_bot0_add, [l2_bot0_cv2_silu_out, l2_cv1_silu_out]).
op_kind(l2_cat, concat).
op_output(l2_cat, l2_cat_out).
op_inputs(l2_cat, [l2_bot0_add_out, l2_cv2_silu_out]).
op_kind(l2_cv3_conv, conv2d).
op_output(l2_cv3_conv, l2_cv3_conv_out).
op_inputs(l2_cv3_conv, [l2_cat_out]).
op_attr(l2_cv3_conv, kernel_size, 1).
op_attr(l2_cv3_conv, in_channels, 32).
op_attr(l2_cv3_conv, out_channels, 32).
op_kind(l2_cv3_bn, batchnorm).
op_output(l2_cv3_bn, l2_cv3_bn_out).
op_inputs(l2_cv3_bn, [l2_cv3_conv_out]).
op_kind(l2_cv3_silu, silu).
op_output(l2_cv3_silu, l2_cv3_silu_out).
op_inputs(l2_cv3_silu, [l2_cv3_bn_out]).
op_kind(l3_conv, conv2d).
op_output(l3_conv, l3_conv_out).
op_inputs(l3_conv, [l2_cv3_silu_out]).
op_attr(l3_conv, kernel_size, 3).
op_attr(l3_conv, in_channels, 32).
op_attr(l3_conv, out_channels, 64).
op_kind(l3_bn, batchnorm).
op_output(l3_bn, l3_bn_out).
op_inputs(l3_bn, [l3_conv_out]).
op_kind(l3_silu, silu).
op_output(l3_silu, l3_silu_out).
op_inputs(l3_silu, [l3_bn_out]).
op_kind(l4_cv1_conv, conv2d).
op_output(l4_cv1_conv, l4_cv1_conv_out).
op_inputs(l4_cv1_conv, [l3_silu_out]).
op_attr(l4_cv1_conv, kernel_size, 1).
op_attr(l4_cv1_conv, in_channels, 64).
op_attr(l4_cv1_conv, out_channels, 32).
op_kind(l4_cv1_bn, batchnorm).
op_output(l4_cv1_bn, l4_cv1_bn_out).
op_inputs(l4_cv1_bn, [l4_cv1_conv_out]).
op_kind(l4_cv1_silu, silu).
op_output(l4_cv1_silu, l4_cv1_silu_out).
op_inputs(l4_cv1_silu, [l4_cv1_bn_out]).
op_kind(l4_cv2_conv, conv2d).
op_output(l4_cv2_conv, l4_cv2_conv_out).
op_inputs(l4_cv2_conv, [l3_silu_out]).
op_attr(l4_cv2_conv, kernel_size, 1).
op_attr(l4_cv2_conv, in_channels, 64).
op_attr(l4_cv2_conv, out_channels, 32).
op_kind(l4_cv2_bn, batchnorm).
op_output(l4_cv2_bn, l4_cv2_bn_out).
op_inputs(l4_cv2_bn, [l4_cv2_conv_out]).
op_kind(l4_cv2_silu, silu).
op_output(l4_cv2_silu, l4_cv2_silu_out).
op_inputs(l4_cv2_silu, [l4_cv2_bn_out]).
op_kind(l4_bot0_cv1_conv, conv2d).
op_output(l4_bot0_cv1_conv, l4_bot0_cv1_conv_out).
op_inputs(l4_bot0_cv1_conv, [l4_cv1_silu_out]).
op_attr(l4_bot0_cv1_conv, kernel_size, 1).
op_attr(l4_bot0_cv1_conv, in_channels, 32).
op_attr(l4_bot0_cv1_conv, out_channels, 32).
op_kind(l4_bot0_cv1_bn, batchnorm).
op_output(l4_bot0_cv1_bn, l4_bot0_cv1_bn_out).
op_inputs(l4_bot0_cv1_bn, [l4_bot0_cv1_conv_out]).
op_kind(l4_bot0_cv1_silu, silu).
op_output(l4_bot0_cv1_silu, l4_bot0_cv1_silu_out).
op_inputs(l4_bot0_cv1_silu, [l4_bot0_cv1_bn_out]).
op_kind(l4_bot0_cv2_conv, conv2d).
op_output(l4_bot0_cv2_conv, l4_bot0_cv2_conv_out).
op_inputs(l4_bot0_cv2_conv, [l4_bot0_cv1_silu_out]).
op_attr(l4_bot0_cv2_conv, kernel_size, 3).
op_attr(l4_bot0_cv2_conv, in_channels, 32).
op_attr(l4_bot0_cv2_conv, out_channels, 32).
op_kind(l4_bot0_cv2_bn, batchnorm).
op_output(l4_bot0_cv2_bn, l4_bot0_cv2_bn_out).
op_inputs(l4_bot0_cv2_bn, [l4_bot0_cv2_conv_out]).
op_kind(l4_bot0_cv2_silu, silu).
op_output(l4_bot0_cv2_silu, l4_bot0_cv2_silu_out).
op_inputs(l4_bot0_cv2_silu, [l4_bot0_cv2_bn_out]).
op_kind(l4_bot0_add, add).
op_output(l4_bot0_add, l4_bot0_add_out).
op_inputs(l4_bot0_add, [l4_bot0_cv2_silu_out, l4_cv1_silu_out]).
op_kind(l4_bot1_cv1_conv, conv2d).
op_output(l4_bot1_cv1_conv, l4_bot1_cv1_conv_out).
op_inputs(l4_bot1_cv1_conv, [l4_bot0_add_out]).
op_attr(l4_bot1_cv1_conv, kernel_size, 1).
op_attr(l4_bot1_cv1_conv, in_channels, 32).
op_attr(l4_bot1_cv1_conv, out_channels, 32).
op_kind(l4_bot1_cv1_bn, batchnorm).
op_output(l4_bot1_cv1_bn, l4_bot1_cv1_bn_out).
op_inputs(l4_bot1_cv1_bn, [l4_bot1_cv1_conv_out]).
op_kind(l4_bot1_cv1_silu, silu).
op_output(l4_bot1_cv1_silu, l4_bot1_cv1_silu_out).
op_inputs(l4_bot1_cv1_silu, [l4_bot1_cv1_bn_out]).
op_kind(l4_bot1_cv2_conv, conv2d).
op_output(l4_bot1_cv2_conv, l4_bot1_cv2_conv_out).
op_inputs(l4_bot1_cv2_conv, [l4_bot1_cv1_silu_out]).
op_attr(l4_bot1_cv2_conv, kernel_size, 3).
op_attr(l4_bot1_cv2_conv, in_channels, 32).
op_attr(l4_bot1_cv2_conv, out_channels, 32).
op_kind(l4_bot1_cv2_bn, batchnorm).
op_output(l4_bot1_cv2_bn, l4_bot1_cv2_bn_out).
op_inputs(l4_bot1_cv2_bn, [l4_bot1_cv2_conv_out]).
op_kind(l4_bot1_cv2_silu, silu).
op_output(l4_bot1_cv2_silu, l4_bot1_cv2_silu_out).
op_inputs(l4_bot1_cv2_silu, [l4_bot1_cv2_bn_out]).
op_kind(l4_bot1_add, add).
op_output(l4_bot1_add, l4_bot1_add_out).
op_inputs(l4_bot1_add, [l4_bot1_cv2_silu_out, l4_bot0_add_out]).
op_kind(l4_cat, concat).
op_output(l4_cat, l4_cat_out).
op_inputs(l4_cat, [l4_bot1_add_out, l4_cv2_silu_out]).
op_kind(l4_cv3_conv, conv2d).
op_output(l4_cv3_conv, l4_cv3_conv_out).
op_inputs(l4_cv3_conv, [l4_cat_out]).
op_attr(l4_cv3_conv, kernel_size, 1).
op_attr(l4_cv3_conv, in_channels, 64).
op_attr(l4_cv3_conv, out_channels, 64).
op_kind(l4_cv3_bn, batchnorm).
op_output(l4_cv3_bn, l4_cv3_bn_out).
op_inputs(l4_cv3_bn, [l4_cv3_conv_out]).
op_kind(l4_cv3_silu, silu).
op_output(l4_cv3_silu, l4_cv3_silu_out).
op_inputs(l4_cv3_silu, [l4_cv3_bn_out]).
op_kind(l5_conv, conv2d).
op_output(l5_conv, l5_conv_out).
op_inputs(l5_conv, [l4_cv3_silu_out]).
op_attr(l5_conv, kernel_size, 3).
op_attr(l5_conv, in_channels, 64).
op_attr(l5_conv, out_channels, 128).
op_kind(l5_bn, batchnorm).
op_output(l5_bn, l5_bn_out).
op_inputs(l5_bn, [l5_conv_out]).
op_kind(l5_silu, silu).
op_output(l5_silu, l5_silu_out).
op_inputs(l5_silu, [l5_bn_out]).
op_kind(l6_cv1_conv, conv2d).
op_output(l6_cv1_conv, l6_cv1_conv_out).
op_inputs(l6_cv1_conv, [l5_silu_out]).
op_attr(l6_cv1_conv, kernel_size, 1).
op_attr(l6_cv1_conv, in_channels, 128).
op_attr(l6_cv1_conv, out_channels, 64).
op_kind(l6_cv1_bn, batchnorm).
op_output(l6_cv1_bn, l6_cv1_bn_out).
op_inputs(l6_cv1_bn, [l6_cv1_conv_out]).
op_kind(l6_cv1_silu, silu).
op_output(l6_cv1_silu, l6_cv1_silu_out).
op_inputs(l6_cv1_silu, [l6_cv1_bn_out]).
op_kind(l6_cv2_conv, conv2d).
op_output(l6_cv2_conv, l6_cv2_conv_out).
op_inputs(l6_cv2_conv, [l5_silu_out]).
op_attr(l6_cv2_conv, kernel_size, 1).
op_attr(l6_cv2_conv, in_channels, 128).
op_attr(l6_cv2_conv, out_channels, 64).
op_kind(l6_cv2_bn, batchnorm).
op_output(l6_cv2_bn, l6_cv2_bn_out).
op_inputs(l6_cv2_bn, [l6_cv2_conv_out]).
op_kind(l6_cv2_silu, silu).
op_output(l6_cv2_silu, l6_cv2_silu_out).
op_inputs(l6_cv2_silu, [l6_cv2_bn_out]).
op_kind(l6_bot0_cv1_conv, conv2d).
op_output(l6_bot0_cv1_conv, l6_bot0_cv1_conv_out).
op_inputs(l6_bot0_cv1_conv, [l6_cv1_silu_out]).
op_attr(l6_bot0_cv1_conv, kernel_size, 1).
op_attr(l6_bot0_cv1_conv, in_channels, 64).
op_attr(l6_bot0_cv1_conv, out_channels, 64).
op_kind(l6_bot0_cv1_bn, batchnorm).
op_output(l6_bot0_cv1_bn, l6_bot0_cv1_bn_out).
op_inputs(l6_bot0_cv1_bn, [l6_bot0_cv1_conv_out]).
op_kind(l6_bot0_cv1_silu, silu).
op_output(l6_bot0_cv1_silu, l6_bot0_cv1_silu_out).
op_inputs(l6_bot0_cv1_silu, [l6_bot0_cv1_bn_out]).
op_kind(l6_bot0_cv2_conv, conv2d).
op_output(l6_bot0_cv2_conv, l6_bot0_cv2_conv_out).
op_inputs(l6_bot0_cv2_conv, [l6_bot0_cv1_silu_out]).
op_attr(l6_bot0_cv2_conv, kernel_size, 3).
op_attr(l6_bot0_cv2_conv, in_channels, 64).
op_attr(l6_bot0_cv2_conv, out_channels, 64).
op_kind(l6_bot0_cv2_bn, batchnorm).
op_output(l6_bot0_cv2_bn, l6_bot0_cv2_bn_out).
op_inputs(l6_bot0_cv2_bn, [l6_bot0_cv2_conv_out]).
op_kind(l6_bot0_cv2_silu, silu).
op_output(l6_bot0_cv2_silu, l6_bot0_cv2_silu_out).
op_inputs(l6_bot0_cv2_silu, [l6_bot0_cv2_bn_out]).
op_kind(l6_bot0_add, add).
op_output(l6_bot0_add, l6_bot0_add_out).
op_inputs(l6_bot0_add, [l6_bot0_cv2_silu_out, l6_cv1_silu_out]).
op_kind(l6_bot1_cv1_conv, conv2d).
op_output(l6_bot1_cv1_conv, l6_bot1_cv1_conv_out).
op_inputs(l6_bot1_cv1_conv, [l6_bot0_add_out]).
op_attr(l6_bot1_cv1_conv, kernel_size, 1).
op_attr(l6_bot1_cv1_conv, in_channels, 64).
op_attr(l6_bot1_cv1_conv, out_channels, 64).
op_kind(l6_bot1_cv1_bn, batchnorm).
op_output(l6_bot1_cv1_bn, l6_bot1_cv1_bn_out).
op_inputs(l6_bot1_cv1_bn, [l6_bot1_cv1_conv_out]).
op_kind(l6_bot1_cv1_silu, silu).
op_output(l6_bot1_cv1_silu, l6_bot1_cv1_silu_out).
op_inputs(l6_bot1_cv1_silu, [l6_bot1_cv1_bn_out]).
op_kind(l6_bot1_cv2_conv, conv2d).
op_output(l6_bot1_cv2_conv, l6_bot1_cv2_conv_out).
op_inputs(l6_bot1_cv2_conv, [l6_bot1_cv1_silu_out]).
op_attr(l6_bot1_cv2_conv, kernel_size, 3).
op_attr(l6_bot1_cv2_conv, in_channels, 64).
op_attr(l6_bot1_cv2_conv, out_channels, 64).
op_kind(l6_bot1_cv2_bn, batchnorm).
op_output(l6_bot1_cv2_bn, l6_bot1_cv2_bn_out).
op_inputs(l6_bot1_cv2_bn, [l6_bot1_cv2_conv_out]).
op_kind(l6_bot1_cv2_silu, silu).
op_output(l6_bot1_cv2_silu, l6_bot1_cv2_silu_out).
op_inputs(l6_bot1_cv2_silu, [l6_bot1_cv2_bn_out]).
op_kind(l6_bot1_add, add).
op_output(l6_bot1_add, l6_bot1_add_out).
op_inputs(l6_bot1_add, [l6_bot1_cv2_silu_out, l6_bot0_add_out]).
op_kind(l6_bot2_cv1_conv, conv2d).
op_output(l6_bot2_cv1_conv, l6_bot2_cv1_conv_out).
op_inputs(l6_bot2_cv1_conv, [l6_bot1_add_out]).
op_attr(l6_bot2_cv1_conv, kernel_size, 1).
op_attr(l6_bot2_cv1_conv, in_channels, 64).
op_attr(l6_bot2_cv1_conv, out_channels, 64).
op_kind(l6_bot2_cv1_bn, batchnorm).
op_output(l6_bot2_cv1_bn, l6_bot2_cv1_bn_out).
op_inputs(l6_bot2_cv1_bn, [l6_bot2_cv1_conv_out]).
op_kind(l6_bot2_cv1_silu, silu).
op_output(l6_bot2_cv1_silu, l6_bot2_cv1_silu_out).
op_inputs(l6_bot2_cv1_silu, [l6_bot2_cv1_bn_out]).
op_kind(l6_bot2_cv2_conv, conv2d).
op_output(l6_bot2_cv2_conv, l6_bot2_cv2_conv_out).
op_inputs(l6_bot2_cv2_conv, [l6_bot2_cv1_silu_out]).
op_attr(l6_bot2_cv2_conv, kernel_size, 3).
op_attr(l6_bot2_cv2_conv, in_channels, 64).
op_attr(l6_bot2_cv2_conv, out_channels, 64).
op_kind(l6_bot2_cv2_bn, batchnorm).
op_output(l6_bot2_cv2_bn, l6_bot2_cv2_bn_out).
op_inputs(l6_bot2_cv2_bn, [l6_bot2_cv2_conv_out]).
op_kind(l6_bot2_cv2_silu, silu).
op_output(l6_bot2_cv2_silu, l6_bot2_cv2_silu_out).
op_inputs(l6_bot2_cv2_silu, [l6_bot2_cv2_bn_out]).
op_kind(l6_bot2_add, add).
op_output(l6_bot2_add, l6_bot2_add_out).
op_inputs(l6_bot2_add, [l6_bot2_cv2_silu_out, l6_bot1_add_out]).
op_kind(l6_cat, concat).
op_output(l6_cat, l6_cat_out).
op_inputs(l6_cat, [l6_bot2_add_out, l6_cv2_silu_out]).
op_kind(l6_cv3_conv, conv2d).
op_output(l6_cv3_conv, l6_cv3_conv_out).
op_inputs(l6_cv3_conv, [l6_cat_out]).
op_attr(l6_cv3_conv, kernel_size, 1).
op_attr(l6_cv3_conv, in_channels, 128).
op_attr(l6_cv3_conv, out_channels, 128).
op_kind(l6_cv3_bn, batchnorm).
op_output(l6_cv3_bn, l6_cv3_bn_out).
op_inputs(l6_cv3_bn, [l6_cv3_conv_out]).
op_kind(l6_cv3_silu, silu).
op_output(l6_cv3_silu, l6_cv3_silu_out).
op_inputs(l6_cv3_silu, [l6_cv3_bn_out]).
op_kind(l7_conv, conv2d).
op_output(l7_conv, l7_conv_out).
op_inputs(l7_conv, [l6_cv3_silu_out]).
op_attr(l7_conv, kernel_size, 3).
op_attr(l7_conv, in_channels, 128).
op_attr(l7_conv, out_channels, 256).
op_kind(l7_bn, batchnorm).
op_output(l7_bn, l7_bn_out).
op_inputs(l7_bn, [l7_conv_out]).
op_kind(l7_silu, silu).
op_output(l7_silu, l7_silu_out).
op_inputs(l7_silu, [l7_bn_out]).
op_kind(l8_cv1_conv, conv2d).
op_output(l8_cv1_conv, l8_cv1_conv_out).
op_inputs(l8_cv1_conv, [l7_silu_out]).
op_attr(l8_cv1_conv, kernel_size, 1).
op_attr(l8_cv1_conv, in_channels, 256).
op_attr(l8_cv1_conv, out_channels, 128).
op_kind(l8_cv1_bn, batchnorm).
op_output(l8_cv1_bn, l8_cv1_bn_out).
op_inputs(l8_cv1_bn, [l8_cv1_conv_out]).
op_kind(l8_cv1_silu, silu).
op_output(l8_cv1_silu, l8_cv1_silu_out).
op_inputs(l8_cv1_silu, [l8_cv1_bn_out]).
op_kind(l8_cv2_conv, conv2d).
op_output(l8_cv2_conv, l8_cv2_conv_out).
op_inputs(l8_cv2_conv, [l7_silu_out]).
op_attr(l8_cv2_conv, kernel_size, 1).
op_attr(l8_cv2_conv, in_channels, 256).
op_attr(l8_cv2_conv, out_channels, 128).
op_kind(l8_cv2_bn, batchnorm).
op_output(l8_cv2_bn, l8_cv2_bn_out).
op_inputs(l8_cv2_bn, [l8_cv2_conv_out]).
op_kind(l8_cv2_silu, silu).
op_output(l8_cv2_silu, l8_cv2_silu_out).
op_inputs(l8_cv2_silu, [l8_cv2_bn_out]).
op_kind(l8_bot0_cv1_conv, conv2d).
op_output(l8_bot0_cv1_conv, l8_bot0_cv1_conv_out).
op_inputs(l8_bot0_cv1_conv, [l8_cv1_silu_out]).
op_attr(l8_bot0_cv1_conv, kernel_size, 1).
op_attr(l8_bot0_cv1_conv, in_channels, 128).
op_attr(l8_bot0_cv1_conv, out_channels, 128).
op_kind(l8_bot0_cv1_bn, batchnorm).
op_output(l8_bot0_cv1_bn, l8_bot0_cv1_bn_out).
op_inputs(l8_bot0_cv1_bn, [l8_bot0_cv1_conv_out]).
op_kind(l8_bot0_cv1_silu, silu).
op_output(l8_bot0_cv1_silu, l8_bot0_cv1_silu_out).
op_inputs(l8_bot0_cv1_silu, [l8_bot0_cv1_bn_out]).
op_kind(l8_bot0_cv2_conv, conv2d).
op_output(l8_bot0_cv2_conv, l8_bot0_cv2_conv_out).
op_inputs(l8_bot0_cv2_conv, [l8_bot0_cv1_silu_out]).
op_attr(l8_bot0_cv2_conv, kernel_size, 3).
op_attr(l8_bot0_cv2_conv, in_channels, 128).
op_attr(l8_bot0_cv2_conv, out_channels, 128).
op_kind(l8_bot0_cv2_bn, batchnorm).
op_output(l8_bot0_cv2_bn, l8_bot0_cv2_bn_out).
op_inputs(l8_bot0_cv2_bn, [l8_bot0_cv2_conv_out]).
op_kind(l8_bot0_cv2_silu, silu).
op_output(l8_bot0_cv2_silu, l8_bot0_cv2_silu_out).
op_inputs(l8_bot0_cv2_silu, [l8_bot0_cv2_bn_out]).
op_kind(l8_bot0_add, add).
op_output(l8_bot0_add, l8_bot0_add_out).
op_inputs(l8_bot0_add, [l8_bot0_cv2_silu_out, l8_cv1_silu_out]).
op_kind(l8_cat, concat).
op_output(l8_cat, l8_cat_out).
op_inputs(l8_cat, [l8_bot0_add_out, l8_cv2_silu_out]).
op_kind(l8_cv3_conv, conv2d).
op_output(l8_cv3_conv, l8_cv3_conv_out).
op_inputs(l8_cv3_conv, [l8_cat_out]).
op_attr(l8_cv3_conv, kernel_size, 1).
op_attr(l8_cv3_conv, in_channels, 256).
op_attr(l8_cv3_conv, out_channels, 256).
op_kind(l8_cv3_bn, batchnorm).
op_output(l8_cv3_bn, l8_cv3_bn_out).
op_inputs(l8_cv3_bn, [l8_cv3_conv_out]).
op_kind(l8_cv3_silu, silu).
op_output(l8_cv3_silu, l8_cv3_silu_out).
op_inputs(l8_cv3_silu, [l8_cv3_bn_out]).
op_kind(l9_cv1_conv, conv2d).
op_output(l9_cv1_conv, l9_cv1_conv_out).
op_inputs(l9_cv1_conv, [l8_cv3_silu_out]).
op_attr(l9_cv1_conv, kernel_size, 1).
op_attr(l9_cv1_conv, in_channels, 256).
op_attr(l9_cv1_conv, out_channels, 128).
op_kind(l9_cv1_bn, batchnorm).
op_output(l9_cv1_bn, l9_cv1_bn_out).
op_inputs(l9_cv1_bn, [l9_cv1_conv_out]).
op_kind(l9_cv1_silu, silu).
op_output(l9_cv1_silu, l9_cv1_silu_out).
op_inputs(l9_cv1_silu, [l9_cv1_bn_out]).
op_kind(l9_mp0, maxpool).
op_output(l9_mp0, l9_mp0_out).
op_inputs(l9_mp0, [l9_cv1_silu_out]).
op_kind(l9_mp1, maxpool).
op_output(l9_mp1, l9_mp1_out).
op_inputs(l9_mp1, [l9_mp0_out]).
op_kind(l9_mp2, maxpool).
op_output(l9_mp2, l9_mp2_out).
op_inputs(l9_mp2, [l9_mp1_out]).
op_kind(l9_cat, concat).
op_output(l9_cat, l9_cat_out).
op_inputs(l9_cat, [l9_cv1_silu_out, l9_mp0_out, l9_mp1_out, l9_mp2_out]).
op_kind(l9_cv2_conv, conv2d).
op_output(l9_cv2_conv, l9_cv2_conv_out).
op_inputs(l9_cv2_conv, [l9_cat_out]).
op_attr(l9_cv2_conv, kernel_size, 1).
op_attr(l9_cv2_conv, in_channels, 512).
op_attr(l9_cv2_conv, out_channels, 256).
op_kind(l9_cv2_bn, batchnorm).
op_output(l9_cv2_bn, l9_cv2_bn_out).
op_inputs(l9_cv2_bn, [l9_cv2_conv_out]).
op_kind(l9_cv2_silu, silu).
op_output(l9_cv2_silu, l9_cv2_silu_out).
op_inputs(l9_cv2_silu, [l9_cv2_bn_out]).
op_kind(l10_conv, conv2d).
op_output(l10_conv, l10_conv_out).
op_inputs(l10_conv, [l9_cv2_silu_out]).
op_attr(l10_conv, kernel_size, 1).
op_attr(l10_conv, in_channels, 256).
op_attr(l10_conv, out_channels, 128).
op_kind(l10_bn, batchnorm).
op_output(l10_bn, l10_bn_out).
op_inputs(l10_bn, [l10_conv_out]).
op_kind(l10_silu, silu).
op_output(l10_silu, l10_silu_out).
op_inputs(l10_silu, [l10_bn_out]).
op_kind(l11_upsample, upsample).
op_output(l11_upsample, l11_up_out).
op_inputs(l11_upsample, [l10_silu_out]).
op_kind(l12_concat, concat).
op_output(l12_concat, l12_concat_out).
op_inputs(l12_concat, [l11_up_out, l6_cv3_silu_out]).
op_kind(l13_cv1_conv, conv2d).
op_output(l13_cv1_conv, l13_cv1_conv_out).
op_inputs(l13_cv1_conv, [l12_concat_out]).
op_attr(l13_cv1_conv, kernel_size, 1).
op_attr(l13_cv1_conv, in_channels, 256).
op_attr(l13_cv1_conv, out_channels, 64).
op_kind(l13_cv1_bn, batchnorm).
op_output(l13_cv1_bn, l13_cv1_bn_out).
op_inputs(l13_cv1_bn, [l13_cv1_conv_out]).
op_kind(l13_cv1_silu, silu).
op_output(l13_cv1_silu, l13_cv1_silu_out).
op_inputs(l13_cv1_silu, [l13_cv1_bn_out]).
op_kind(l13_cv2_conv, conv2d).
op_output(l13_cv2_conv, l13_cv2_conv_out).
op_inputs(l13_cv2_conv, [l12_concat_out]).
op_attr(l13_cv2_conv, kernel_size, 1).
op_attr(l13_cv2_conv, in_channels, 256).
op_attr(l13_cv2_conv, out_channels, 64).
op_kind(l13_cv2_bn, batchnorm).
op_output(l13_cv2_bn, l13_cv2_bn_out).
op_inputs(l13_cv2_bn, [l13_cv2_conv_out]).
op_kind(l13_cv2_silu, silu).
op_output(l13_cv2_silu, l13_cv2_silu_out).
op_inputs(l13_cv2_silu, [l13_cv2_bn_out]).
op_kind(l13_bot0_cv1_conv, conv2d).
op_output(l13_bot0_cv1_conv, l13_bot0_cv1_conv_out).
op_inputs(l13_bot0_cv1_conv, [l13_cv1_silu_out]).
op_attr(l13_bot0_cv1_conv, kernel_size, 1).
op_attr(l13_bot0_cv1_conv, in_channels, 64).
op_attr(l13_bot0_cv1_conv, out_channels, 64).
op_kind(l13_bot0_cv1_bn, batchnorm).
op_output(l13_bot0_cv1_bn, l13_bot0_cv1_bn_out).
op_inputs(l13_bot0_cv1_bn, [l13_bot0_cv1_conv_out]).
op_kind(l13_bot0_cv1_silu, silu).
op_output(l13_bot0_cv1_silu, l13_bot0_cv1_silu_out).
op_inputs(l13_bot0_cv1_silu, [l13_bot0_cv1_bn_out]).
op_kind(l13_bot0_cv2_conv, conv2d).
op_output(l13_bot0_cv2_conv, l13_bot0_cv2_conv_out).
op_inputs(l13_bot0_cv2_conv, [l13_bot0_cv1_silu_out]).
op_attr(l13_bot0_cv2_conv, kernel_size, 3).
op_attr(l13_bot0_cv2_conv, in_channels, 64).
op_attr(l13_bot0_cv2_conv, out_channels, 64).
op_kind(l13_bot0_cv2_bn, batchnorm).
op_output(l13_bot0_cv2_bn, l13_bot0_cv2_bn_out).
op_inputs(l13_bot0_cv2_bn, [l13_bot0_cv2_conv_out]).
op_kind(l13_bot0_cv2_silu, silu).
op_output(l13_bot0_cv2_silu, l13_bot0_cv2_silu_out).
op_inputs(l13_bot0_cv2_silu, [l13_bot0_cv2_bn_out]).
op_kind(l13_cat, concat).
op_output(l13_cat, l13_cat_out).
op_inputs(l13_cat, [l13_bot0_cv2_silu_out, l13_cv2_silu_out]).
op_kind(l13_cv3_conv, conv2d).
op_output(l13_cv3_conv, l13_cv3_conv_out).
op_inputs(l13_cv3_conv, [l13_cat_out]).
op_attr(l13_cv3_conv, kernel_size, 1).
op_attr(l13_cv3_conv, in_channels, 128).
op_attr(l13_cv3_conv, out_channels, 128).
op_kind(l13_cv3_bn, batchnorm).
op_output(l13_cv3_bn, l13_cv3_bn_out).
op_inputs(l13_cv3_bn, [l13_cv3_conv_out]).
op_kind(l13_cv3_silu, silu).
op_output(l13_cv3_silu, l13_cv3_silu_out).
op_inputs(l13_cv3_silu, [l13_cv3_bn_out]).
op_kind(l14_conv, conv2d).
op_output(l14_conv, l14_conv_out).
op_inputs(l14_conv, [l13_cv3_silu_out]).
op_attr(l14_conv, kernel_size, 1).
op_attr(l14_conv, in_channels, 128).
op_attr(l14_conv, out_channels, 64).
op_kind(l14_bn, batchnorm).
op_output(l14_bn, l14_bn_out).
op_inputs(l14_bn, [l14_conv_out]).
op_kind(l14_silu, silu).
op_output(l14_silu, l14_silu_out).
op_inputs(l14_silu, [l14_bn_out]).
op_kind(l15_upsample, upsample).
op_output(l15_upsample, l15_up_out).
op_inputs(l15_upsample, [l14_silu_out]).
op_kind(l16_concat, concat).
op_output(l16_concat, l16_concat_out).
op_inputs(l16_concat, [l15_up_out, l4_cv3_silu_out]).
op_kind(l17_cv1_conv, conv2d).
op_output(l17_cv1_conv, l17_cv1_conv_out).
op_inputs(l17_cv1_conv, [l16_concat_out]).
op_attr(l17_cv1_conv, kernel_size, 1).
op_attr(l17_cv1_conv, in_channels, 128).
op_attr(l17_cv1_conv, out_channels, 32).
op_kind(l17_cv1_bn, batchnorm).
op_output(l17_cv1_bn, l17_cv1_bn_out).
op_inputs(l17_cv1_bn, [l17_cv1_conv_out]).
op_kind(l17_cv1_silu, silu).
op_output(l17_cv1_silu, l17_cv1_silu_out).
op_inputs(l17_cv1_silu, [l17_cv1_bn_out]).
op_kind(l17_cv2_conv, conv2d).
op_output(l17_cv2_conv, l17_cv2_conv_out).
op_inputs(l17_cv2_conv, [l16_concat_out]).
op_attr(l17_cv2_conv, kernel_size, 1).
op_attr(l17_cv2_conv, in_channels, 128).
op_attr(l17_cv2_conv, out_channels, 32).
op_kind(l17_cv2_bn, batchnorm).
op_output(l17_cv2_bn, l17_cv2_bn_out).
op_inputs(l17_cv2_bn, [l17_cv2_conv_out]).
op_kind(l17_cv2_silu, silu).
op_output(l17_cv2_silu, l17_cv2_silu_out).
op_inputs(l17_cv2_silu, [l17_cv2_bn_out]).
op_kind(l17_bot0_cv1_conv, conv2d).
op_output(l17_bot0_cv1_conv, l17_bot0_cv1_conv_out).
op_inputs(l17_bot0_cv1_conv, [l17_cv1_silu_out]).
op_attr(l17_bot0_cv1_conv, kernel_size, 1).
op_attr(l17_bot0_cv1_conv, in_channels, 32).
op_attr(l17_bot0_cv1_conv, out_channels, 32).
op_kind(l17_bot0_cv1_bn, batchnorm).
op_output(l17_bot0_cv1_bn, l17_bot0_cv1_bn_out).
op_inputs(l17_bot0_cv1_bn, [l17_bot0_cv1_conv_out]).
op_kind(l17_bot0_cv1_silu, silu).
op_output(l17_bot0_cv1_silu, l17_bot0_cv1_silu_out).
op_inputs(l17_bot0_cv1_silu, [l17_bot0_cv1_bn_out]).
op_kind(l17_bot0_cv2_conv, conv2d).
op_output(l17_bot0_cv2_conv, l17_bot0_cv2_conv_out).
op_inputs(l17_bot0_cv2_conv, [l17_bot0_cv1_silu_out]).
op_attr(l17_bot0_cv2_conv, kernel_size, 3).
op_attr(l17_bot0_cv2_conv, in_channels, 32).
op_attr(l17_bot0_cv2_conv, out_channels, 32).
op_kind(l17_bot0_cv2_bn, batchnorm).
op_output(l17_bot0_cv2_bn, l17_bot0_cv2_bn_out).
op_inputs(l17_bot0_cv2_bn, [l17_bot0_cv2_conv_out]).
op_kind(l17_bot0_cv2_silu, silu).
op_output(l17_bot0_cv2_silu, l17_bot0_cv2_silu_out).
op_inputs(l17_bot0_cv2_silu, [l17_bot0_cv2_bn_out]).
op_kind(l17_cat, concat).
op_output(l17_cat, l17_cat_out).
op_inputs(l17_cat, [l17_bot0_cv2_silu_out, l17_cv2_silu_out]).
op_kind(l17_cv3_conv, conv2d).
op_output(l17_cv3_conv, l17_cv3_conv_out).
op_inputs(l17_cv3_conv, [l17_cat_out]).
op_attr(l17_cv3_conv, kernel_size, 1).
op_attr(l17_cv3_conv, in_channels, 64).
op_attr(l17_cv3_conv, out_channels, 64).
op_kind(l17_cv3_bn, batchnorm).
op_output(l17_cv3_bn, l17_cv3_bn_out).
op_inputs(l17_cv3_bn, [l17_cv3_conv_out]).
op_kind(l17_cv3_silu, silu).
op_output(l17_cv3_silu, l17_cv3_silu_out).
op_inputs(l17_cv3_silu, [l17_cv3_bn_out]).
op_kind(l18_conv, conv2d).
op_output(l18_conv, l18_conv_out).
op_inputs(l18_conv, [l17_cv3_silu_out]).
op_attr(l18_conv, kernel_size, 3).
op_attr(l18_conv, in_channels, 64).
op_attr(l18_conv, out_channels, 64).
op_kind(l18_bn, batchnorm).
op_output(l18_bn, l18_bn_out).
op_inputs(l18_bn, [l18_conv_out]).
op_kind(l18_silu, silu).
op_output(l18_silu, l18_silu_out).
op_inputs(l18_silu, [l18_bn_out]).
op_kind(l19_concat, concat).
op_output(l19_concat, l19_concat_out).
op_inputs(l19_concat, [l18_silu_out, l14_silu_out]).
op_kind(l20_cv1_conv, conv2d).
op_output(l20_cv1_conv, l20_cv1_conv_out).
op_inputs(l20_cv1_conv, [l19_concat_out]).
op_attr(l20_cv1_conv, kernel_size, 1).
op_attr(l20_cv1_conv, in_channels, 128).
op_attr(l20_cv1_conv, out_channels, 64).
op_kind(l20_cv1_bn, batchnorm).
op_output(l20_cv1_bn, l20_cv1_bn_out).
op_inputs(l20_cv1_bn, [l20_cv1_conv_out]).
op_kind(l20_cv1_silu, silu).
op_output(l20_cv1_silu, l20_cv1_silu_out).
op_inputs(l20_cv1_silu, [l20_cv1_bn_out]).
op_kind(l20_cv2_conv, conv2d).
op_output(l20_cv2_conv, l20_cv2_conv_out).
op_inputs(l20_cv2_conv, [l19_concat_out]).
op_attr(l20_cv2_conv, kernel_size, 1).
op_attr(l20_cv2_conv, in_channels, 128).
op_attr(l20_cv2_conv, out_channels, 64).
op_kind(l20_cv2_bn, batchnorm).
op_output(l20_cv2_bn, l20_cv2_bn_out).
op_inputs(l20_cv2_bn, [l20_cv2_conv_out]).
op_kind(l20_cv2_silu, silu).
op_output(l20_cv2_silu, l20_cv2_silu_out).
op_inputs(l20_cv2_silu, [l20_cv2_bn_out]).
op_kind(l20_bot0_cv1_conv, conv2d).
op_output(l20_bot0_cv1_conv, l20_bot0_cv1_conv_out).
op_inputs(l20_bot0_cv1_conv, [l20_cv1_silu_out]).
op_attr(l20_bot0_cv1_conv, kernel_size, 1).
op_attr(l20_bot0_cv1_conv, in_channels, 64).
op_attr(l20_bot0_cv1_conv, out_channels, 64).
op_kind(l20_bot0_cv1_bn, batchnorm).
op_output(l20_bot0_cv1_bn, l20_bot0_cv1_bn_out).
op_inputs(l20_bot0_cv1_bn, [l20_bot0_cv1_conv_out]).
op_kind(l20_bot0_cv1_silu, silu).
op_output(l20_bot0_cv1_silu, l20_bot0_cv1_silu_out).
op_inputs(l20_bot0_cv1_silu, [l20_bot0_cv1_bn_out]).
op_kind(l20_bot0_cv2_conv, conv2d).
op_output(l20_bot0_cv2_conv, l20_bot0_cv2_conv_out).
op_inputs(l20_bot0_cv2_conv, [l20_bot0_cv1_silu_out]).
op_attr(l20_bot0_cv2_conv, kernel_size, 3).
op_attr(l20_bot0_cv2_conv, in_channels, 64).
op_attr(l20_bot0_cv2_conv, out_channels, 64).
op_kind(l20_bot0_cv2_bn, batchnorm).
op_output(l20_bot0_cv2_bn, l20_bot0_cv2_bn_out).
op_inputs(l20_bot0_cv2_bn, [l20_bot0_cv2_conv_out]).
op_kind(l20_bot0_cv2_silu, silu).
op_output(l20_bot0_cv2_silu, l20_bot0_cv2_silu_out).
op_inputs(l20_bot0_cv2_silu, [l20_bot0_cv2_bn_out]).
op_kind(l20_cat, concat).
op_output(l20_cat, l20_cat_out).
op_inputs(l20_cat, [l20_bot0_cv2_silu_out, l20_cv2_silu_out]).
op_kind(l20_cv3_conv, conv2d).
op_output(l20_cv3_conv, l20_cv3_conv_out).
op_inputs(l20_cv3_conv, [l20_cat_out]).
op_attr(l20_cv3_conv, kernel_size, 1).
op_attr(l20_cv3_conv, in_channels, 128).
op_attr(l20_cv3_conv, out_channels, 128).
op_kind(l20_cv3_bn, batchnorm).
op_output(l20_cv3_bn, l20_cv3_bn_out).
op_inputs(l20_cv3_bn, [l20_cv3_conv_out]).
op_kind(l20_cv3_silu, silu).
op_output(l20_cv3_silu, l20_cv3_silu_out).
op_inputs(l20_cv3_silu, [l20_cv3_bn_out]).
op_kind(l21_conv, conv2d).
op_output(l21_conv, l21_conv_out).
op_inputs(l21_conv, [l20_cv3_silu_out]).
op_attr(l21_conv, kernel_size, 3).
op_attr(l21_conv, in_channels, 128).
op_attr(l21_conv, out_channels, 128).
op_kind(l21_bn, batchnorm).
op_output(l21_bn, l21_bn_out).
op_inputs(l21_bn, [l21_conv_out]).
op_kind(l21_silu, silu).
op_output(l21_silu, l21_silu_out).
op_inputs(l21_silu, [l21_bn_out]).
op_kind(l22_concat, concat).
op_output(l22_concat, l22_concat_out).
op_inputs(l22_concat, [l21_silu_out, l10_silu_out]).
op_kind(l23_cv1_conv, conv2d).
op_output(l23_cv1_conv, l23_cv1_conv_out).
op_inputs(l23_cv1_conv, [l22_concat_out]).
op_attr(l23_cv1_conv, kernel_size, 1).
op_attr(l23_cv1_conv, in_channels, 256).
op_attr(l23_cv1_conv, out_channels, 128).
op_kind(l23_cv1_bn, batchnorm).
op_output(l23_cv1_bn, l23_cv1_bn_out).
op_inputs(l23_cv1_bn, [l23_cv1_conv_out]).
op_kind(l23_cv1_silu, silu).
op_output(l23_cv1_silu, l23_cv1_silu_out).
op_inputs(l23_cv1_silu, [l23_cv1_bn_out]).
op_kind(l23_cv2_conv, conv2d).
op_output(l23_cv2_conv, l23_cv2_conv_out).
op_inputs(l23_cv2_conv, [l22_concat_out]).
op_attr(l23_cv2_conv, kernel_size, 1).
op_attr(l23_cv2_conv, in_channels, 256).
op_attr(l23_cv2_conv, out_channels, 128).
op_kind(l23_cv2_bn, batchnorm).
op_output(l23_cv2_bn, l23_cv2_bn_out).
op_inputs(l23_cv2_bn, [l23_cv2_conv_out]).
op_kind(l23_cv2_silu, silu).
op_output(l23_cv2_silu, l23_cv2_silu_out).
op_inputs(l23_cv2_silu, [l23_cv2_bn_out]).
op_kind(l23_bot0_cv1_conv, conv2d).
op_output(l23_bot0_cv1_conv, l23_bot0_cv1_conv_out).
op_inputs(l23_bot0_cv1_conv, [l23_cv1_silu_out]).
op_attr(l23_bot0_cv1_conv, kernel_size, 1).
op_attr(l23_bot0_cv1_conv, in_channels, 128).
op_attr(l23_bot0_cv1_conv, out_channels, 128).
op_kind(l23_bot0_cv1_bn, batchnorm).
op_output(l23_bot0_cv1_bn, l23_bot0_cv1_bn_out).
op_inputs(l23_bot0_cv1_bn, [l23_bot0_cv1_conv_out]).
op_kind(l23_bot0_cv1_silu, silu).
op_output(l23_bot0_cv1_silu, l23_bot0_cv1_silu_out).
op_inputs(l23_bot0_cv1_silu, [l23_bot0_cv1_bn_out]).
op_kind(l23_bot0_cv2_conv, conv2d).
op_output(l23_bot0_cv2_conv, l23_bot0_cv2_conv_out).
op_inputs(l23_bot0_cv2_conv, [l23_bot0_cv1_silu_out]).
op_attr(l23_bot0_cv2_conv, kernel_size, 3).
op_attr(l23_bot0_cv2_conv, in_channels, 128).
op_attr(l23_bot0_cv2_conv, out_channels, 128).
op_kind(l23_bot0_cv2_bn, batchnorm).
op_output(l23_bot0_cv2_bn, l23_bot0_cv2_bn_out).
op_inputs(l23_bot0_cv2_bn, [l23_bot0_cv2_conv_out]).
op_kind(l23_bot0_cv2_silu, silu).
op_output(l23_bot0_cv2_silu, l23_bot0_cv2_silu_out).
op_inputs(l23_bot0_cv2_silu, [l23_bot0_cv2_bn_out]).
op_kind(l23_cat, concat).
op_output(l23_cat, l23_cat_out).
op_inputs(l23_cat, [l23_bot0_cv2_silu_out, l23_cv2_silu_out]).
op_kind(l23_cv3_conv, conv2d).
op_output(l23_cv3_conv, l23_cv3_conv_out).
op_inputs(l23_cv3_conv, [l23_cat_out]).
op_attr(l23_cv3_conv, kernel_size, 1).
op_attr(l23_cv3_conv, in_channels, 256).
op_attr(l23_cv3_conv, out_channels, 256).
op_kind(l23_cv3_bn, batchnorm).
op_output(l23_cv3_bn, l23_cv3_bn_out).
op_inputs(l23_cv3_bn, [l23_cv3_conv_out]).
op_kind(l23_cv3_silu, silu).
op_output(l23_cv3_silu, l23_cv3_silu_out).
op_inputs(l23_cv3_silu, [l23_cv3_bn_out]).
op_kind(det0_conv, conv2d).
op_output(det0_conv, det0_out).
op_inputs(det0_conv, [l17_cv3_silu_out]).
op_kind(det1_conv, conv2d).
op_output(det1_conv, det1_out).
op_inputs(det1_conv, [l20_cv3_silu_out]).
op_kind(det2_conv, conv2d).
op_output(det2_conv, det2_out).
op_inputs(det2_conv, [l23_cv3_silu_out]).
