# CUDA-oxide sm_61 + cmath transcendentals patch
# 
# Three files need changes:
#   1. crates/dialect-mir/src/rust_intrinsics.rs — add cmath placeholder constants
#   2. crates/mir-lower/src/convert/ops/call.rs — add cmath → __nv_* mappings
#   3. crates/rustc-codegen-cuda/src/collector.rs — intercept std::sys::cmath calls
#
# Plus one fix to the compilation pipeline:
#   4. crates/mir-importer/src/pipeline.rs — link libdevice before llc instead of skipping

# ═══════════════════════════════════════════════════════════════
# PATCH 1: rust_intrinsics.rs — add cmath placeholder constants
# ═══════════════════════════════════════════════════════════════
# 
# These functions exist in libdevice but Rust calls them through
# std::sys::cmath (C FFI) instead of core::intrinsics.
# We add placeholder names so the collector can rewrite them.

--- a/crates/dialect-mir/src/rust_intrinsics.rs
+++ b/crates/dialect-mir/src/rust_intrinsics.rs
@@ end of file (after CALLEE_COPYSIGN_F64)
+
+// ── cmath functions not in core::intrinsics ──────────────────
+// Rust's .tanh(), .sinh(), .cosh(), .asin(), .acos(), .atan(),
+// .atan2(), and .erf() go through std::sys::cmath (C FFI) rather
+// than core::intrinsics. We intercept them in the collector and
+// rewrite to these placeholders, which the MIR-to-LLVM lowering
+// then maps to __nv_* libdevice calls.
+
+/// Placeholder for `std::sys::cmath::tanhf` → `__nv_tanhf`
+pub const CALLEE_TANH_F32: &str = placeholder!("tanhf32");
+/// Placeholder for `std::sys::cmath::tanh` → `__nv_tanh`
+pub const CALLEE_TANH_F64: &str = placeholder!("tanhf64");
+/// Placeholder for `std::sys::cmath::sinhf` → `__nv_sinhf`
+pub const CALLEE_SINH_F32: &str = placeholder!("sinhf32");
+/// Placeholder for `std::sys::cmath::sinh` → `__nv_sinh`
+pub const CALLEE_SINH_F64: &str = placeholder!("sinhf64");
+/// Placeholder for `std::sys::cmath::coshf` → `__nv_coshf`
+pub const CALLEE_COSH_F32: &str = placeholder!("coshf32");
+/// Placeholder for `std::sys::cmath::cosh` → `__nv_cosh`
+pub const CALLEE_COSH_F64: &str = placeholder!("coshf64");
+/// Placeholder for `std::sys::cmath::asinf` → `__nv_asinf`
+pub const CALLEE_ASIN_F32: &str = placeholder!("asinf32");
+/// Placeholder for `std::sys::cmath::asin` → `__nv_asin`
+pub const CALLEE_ASIN_F64: &str = placeholder!("asinf64");
+/// Placeholder for `std::sys::cmath::acosf` → `__nv_acosf`
+pub const CALLEE_ACOS_F32: &str = placeholder!("acosf32");
+/// Placeholder for `std::sys::cmath::acos` → `__nv_acos`
+pub const CALLEE_ACOS_F64: &str = placeholder!("acosf64");
+/// Placeholder for `std::sys::cmath::atanf` → `__nv_atanf`
+pub const CALLEE_ATAN_F32: &str = placeholder!("atanf32");
+/// Placeholder for `std::sys::cmath::atan` → `__nv_atan`
+pub const CALLEE_ATAN_F64: &str = placeholder!("atanf64");
+/// Placeholder for `std::sys::cmath::atan2f` → `__nv_atan2f`
+pub const CALLEE_ATAN2_F32: &str = placeholder!("atan2f32");
+/// Placeholder for `std::sys::cmath::atan2` → `__nv_atan2`
+pub const CALLEE_ATAN2_F64: &str = placeholder!("atan2f64");
+/// Placeholder for `std::sys::cmath::erff` → `__nv_erff`
+pub const CALLEE_ERF_F32: &str = placeholder!("erff32");
+/// Placeholder for `std::sys::cmath::erf` → `__nv_erf`
+pub const CALLEE_ERF_F64: &str = placeholder!("erff64");
