// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @shfltest(ptr %out) {
  %v = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float 1.0, i32 16, i32 31)
  store float %v, ptr %out
  ret void
}
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)
