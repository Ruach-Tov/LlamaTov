# ═══════════════════════════════════════════════════════════════
# PATCH 2: call.rs — add cmath → __nv_* mappings
# ═══════════════════════════════════════════════════════════════

--- a/crates/mir-lower/src/convert/ops/call.rs
+++ b/crates/mir-lower/src/convert/ops/call.rs

# Add to enum RustFloatMathIntrinsic (after CopysignF64):

+    // cmath functions (not in core::intrinsics, intercepted from std::sys::cmath)
+    TanhF32,
+    TanhF64,
+    SinhF32,
+    SinhF64,
+    CoshF32,
+    CoshF64,
+    AsinF32,
+    AsinF64,
+    AcosF32,
+    AcosF64,
+    AtanF32,
+    AtanF64,
+    Atan2F32,
+    Atan2F64,
+    ErfF32,
+    ErfF64,

# Add to from_placeholder_callee match arms (after CALLEE_COPYSIGN_F64):

+            rust_intrinsics::CALLEE_TANH_F32 => Some(Self::TanhF32),
+            rust_intrinsics::CALLEE_TANH_F64 => Some(Self::TanhF64),
+            rust_intrinsics::CALLEE_SINH_F32 => Some(Self::SinhF32),
+            rust_intrinsics::CALLEE_SINH_F64 => Some(Self::SinhF64),
+            rust_intrinsics::CALLEE_COSH_F32 => Some(Self::CoshF32),
+            rust_intrinsics::CALLEE_COSH_F64 => Some(Self::CoshF64),
+            rust_intrinsics::CALLEE_ASIN_F32 => Some(Self::AsinF32),
+            rust_intrinsics::CALLEE_ASIN_F64 => Some(Self::AsinF64),
+            rust_intrinsics::CALLEE_ACOS_F32 => Some(Self::AcosF32),
+            rust_intrinsics::CALLEE_ACOS_F64 => Some(Self::AcosF64),
+            rust_intrinsics::CALLEE_ATAN_F32 => Some(Self::AtanF32),
+            rust_intrinsics::CALLEE_ATAN_F64 => Some(Self::AtanF64),
+            rust_intrinsics::CALLEE_ATAN2_F32 => Some(Self::Atan2F32),
+            rust_intrinsics::CALLEE_ATAN2_F64 => Some(Self::Atan2F64),
+            rust_intrinsics::CALLEE_ERF_F32 => Some(Self::ErfF32),
+            rust_intrinsics::CALLEE_ERF_F64 => Some(Self::ErfF64),

# Add to libdevice_name match arms (after CopysignF64):

+            Self::TanhF32 => Ok("__nv_tanhf"),
+            Self::TanhF64 => Ok("__nv_tanh"),
+            Self::SinhF32 => Ok("__nv_sinhf"),
+            Self::SinhF64 => Ok("__nv_sinh"),
+            Self::CoshF32 => Ok("__nv_coshf"),
+            Self::CoshF64 => Ok("__nv_cosh"),
+            Self::AsinF32 => Ok("__nv_asinf"),
+            Self::AsinF64 => Ok("__nv_asin"),
+            Self::AcosF32 => Ok("__nv_acosf"),
+            Self::AcosF64 => Ok("__nv_acos"),
+            Self::AtanF32 => Ok("__nv_atanf"),
+            Self::AtanF64 => Ok("__nv_atan"),
+            Self::Atan2F32 => Ok("__nv_atan2f"),
+            Self::Atan2F64 => Ok("__nv_atan2"),
+            Self::ErfF32 => Ok("__nv_erff"),
+            Self::ErfF64 => Ok("__nv_erf"),

# Add to arg_count match (atan2 takes 2 args):

+            Self::Atan2F32 | Self::Atan2F64 => 2,
