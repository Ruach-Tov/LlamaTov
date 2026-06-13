# ═══════════════════════════════════════════════════════════════
# PATCH 3: collector.rs — intercept std::sys::cmath calls
# ═══════════════════════════════════════════════════════════════
#
# This is the KEY patch. When the collector encounters a call to
# std::sys::cmath::tanhf (or sinf, cosf, etc.), instead of panicking
# with "FORBIDDEN CRATE IN DEVICE CODE", it rewrites the call to
# the corresponding placeholder that the lowering pass handles.

--- a/crates/rustc-codegen-cuda/src/collector.rs
+++ b/crates/rustc-codegen-cuda/src/collector.rs

# In fn should_collect_from_crate, BEFORE the "std" forbidden check,
# add an intercept for std::sys::cmath:

     fn should_collect_from_crate(&self, def_id: DefId) -> CollectDecision {
         // Always collect from local crate
         if def_id.krate == LOCAL_CRATE {
             return CollectDecision::Collect;
         }
 
         let crate_name = self.tcx.crate_name(def_id.krate);
         let name_str = crate_name.as_str();
 
+        // Intercept std::sys::cmath functions — these are C math library
+        // calls (tanhf, sinhf, etc.) that Rust delegates to libm via FFI.
+        // On GPU, we rewrite them to libdevice __nv_* calls via the same
+        // placeholder mechanism used for core::intrinsics math functions.
+        if name_str == "std" {
+            let fn_path = self.tcx.def_path_str(def_id);
+            if let Some(placeholder) = cmath_to_placeholder(&fn_path) {
+                // Rewrite: instead of collecting this std function,
+                // the MIR importer will emit a placeholder call that
+                // the lowering pass converts to __nv_*.
+                return CollectDecision::RewriteToPlaceholder(placeholder);
+            }
+        }
+
         // ... existing code continues (forbidden crate check for std, etc.)

# Add the cmath mapping function (at module level):

+/// Maps `std::sys::cmath::*` function paths to cuda-oxide placeholder names.
+///
+/// These functions exist in NVIDIA's libdevice but Rust calls them through
+/// C FFI (std → libc → libm) instead of core::intrinsics. We intercept
+/// them here and rewrite to the same placeholder mechanism.
+fn cmath_to_placeholder(fn_path: &str) -> Option<&'static str> {
+    // std::sys::cmath::tanhf → __cuda_oxide_rust_intrinsic_tanhf32
+    // The path format varies by Rust version; match on the function name.
+    let fn_name = fn_path.rsplit("::").next()?;
+    match fn_name {
+        "tanhf"  => Some(rust_intrinsics::CALLEE_TANH_F32),
+        "tanh"   => Some(rust_intrinsics::CALLEE_TANH_F64),
+        "sinhf"  => Some(rust_intrinsics::CALLEE_SINH_F32),
+        "sinh"   => Some(rust_intrinsics::CALLEE_SINH_F64),
+        "coshf"  => Some(rust_intrinsics::CALLEE_COSH_F32),
+        "cosh"   => Some(rust_intrinsics::CALLEE_COSH_F64),
+        "asinf"  => Some(rust_intrinsics::CALLEE_ASIN_F32),
+        "asin"   => Some(rust_intrinsics::CALLEE_ASIN_F64),
+        "acosf"  => Some(rust_intrinsics::CALLEE_ACOS_F32),
+        "acos"   => Some(rust_intrinsics::CALLEE_ACOS_F64),
+        "atanf"  => Some(rust_intrinsics::CALLEE_ATAN_F32),
+        "atan"   => Some(rust_intrinsics::CALLEE_ATAN_F64),
+        "atan2f" => Some(rust_intrinsics::CALLEE_ATAN2_F32),
+        "atan2"  => Some(rust_intrinsics::CALLEE_ATAN2_F64),
+        "erff"   => Some(rust_intrinsics::CALLEE_ERF_F32),
+        "erf"    => Some(rust_intrinsics::CALLEE_ERF_F64),
+        _ => None,  // Not a math function we handle — fall through to forbidden
+    }
+}

# Add a new variant to CollectDecision:

 enum CollectDecision {
     Collect,
     SkipIntentional,
     Forbidden { crate_name: String, fn_path: String },
+    /// Rewrite this call to a placeholder that the MIR lowering handles.
+    /// Used for std::sys::cmath functions that map to __nv_* libdevice calls.
+    RewriteToPlaceholder(&'static str),
 }
