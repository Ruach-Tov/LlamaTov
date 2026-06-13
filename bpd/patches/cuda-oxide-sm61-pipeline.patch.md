# ═══════════════════════════════════════════════════════════════
# PATCH 4: pipeline.rs — link libdevice before llc
# ═══════════════════════════════════════════════════════════════
#
# Currently when __nv_* calls are detected, the pipeline SKIPS llc
# and emits NVVM IR instead. This requires nvJitLink at runtime,
# which fails on sm_61 with ModuleNotFound.
#
# The fix: link libdevice.10.bc with the kernel LLVM IR using
# llvm-link, then run opt to internalize/dead-strip, then run llc
# normally to generate PTX. This works on ALL sm architectures.

--- a/crates/mir-importer/src/pipeline.rs
+++ b/crates/mir-importer/src/pipeline.rs

# Replace the block at ~line 341-370 where needs_libdevice triggers skip:

-    let needs_libdevice = module_uses_libdevice(&ctx, module_op_ptr);
-    let emit_nvvm_ir = config.emit_nvvm_ir || needs_libdevice;
-    
-    if needs_libdevice && !config.emit_nvvm_ir {
-        eprintln!(
-            "\n=== Detected CUDA libdevice (`__nv_*`) calls; \
-             auto-emitting NVVM IR (skip llc) ===\n"
-        );
-    }
-    
-    if emit_nvvm_ir {
-        // ... skip llc, emit .ll only ...
-    }

+    let needs_libdevice = module_uses_libdevice(&ctx, module_op_ptr);
+    
+    if config.emit_nvvm_ir {
+        // Explicit --emit-nvvm-ir flag: skip llc, emit .ll only (existing behavior)
+        // ... existing NVVM IR emission code ...
+    } else if needs_libdevice {
+        // Kernel uses libdevice math functions (__nv_*).
+        // Link libdevice.10.bc, optimize, then generate PTX via llc.
+        // This works on ALL sm architectures including sm_61 (Pascal).
+        eprintln!(
+            "\n=== Detected libdevice calls; linking libdevice.10.bc before llc ===\n"
+        );
+        
+        // Step 1: Write kernel LLVM IR to temp file
+        let kernel_ll = ir_path;  // already written above
+        
+        // Step 2: Find libdevice.10.bc
+        let libdevice_path = find_libdevice(config)?;
+        
+        // Step 3: llvm-link kernel.ll + libdevice.10.bc → merged.bc
+        let merged_bc = kernel_ll.with_extension("merged.bc");
+        let llvm_link = find_llvm_tool("llvm-link", config)?;
+        let status = std::process::Command::new(&llvm_link)
+            .arg(&kernel_ll)
+            .arg(&libdevice_path)
+            .arg("-o")
+            .arg(&merged_bc)
+            .status()?;
+        if !status.success() {
+            return Err(anyhow::anyhow!("llvm-link failed"));
+        }
+        
+        // Step 4: opt — internalize everything except kernel entry points,
+        //         then dead-strip unused libdevice functions
+        let opt_bc = kernel_ll.with_extension("opt.bc");
+        let opt = find_llvm_tool("opt", config)?;
+        let public_api = kernel_names.join(",");  // comma-separated kernel names
+        let status = std::process::Command::new(&opt)
+            .arg("-passes=internalize,globalopt,function(instcombine,simplifycfg)")
+            .arg(&format!("-internalize-public-api-list={}", public_api))
+            .arg(&merged_bc)
+            .arg("-o")
+            .arg(&opt_bc)
+            .status()?;
+        if !status.success() {
+            return Err(anyhow::anyhow!("opt failed"));
+        }
+        
+        // Step 5: llc — generate PTX (same as non-libdevice path)
+        let ptx_path = kernel_ll.with_extension("ptx");
+        let llc = find_llc(config)?;
+        let status = std::process::Command::new(&llc)
+            .arg("-march=nvptx64")
+            .arg(&format!("-mcpu={}", config.target))
+            .arg(&opt_bc)
+            .arg("-o")
+            .arg(&ptx_path)
+            .status()?;
+        if !status.success() {
+            return Err(anyhow::anyhow!("llc failed for libdevice-linked kernel"));
+        }
+        
+        // Continue with normal PTX embedding flow
+        // ... (same as the non-libdevice path below)
+    }

# Add helper function to find libdevice.10.bc:

+/// Find libdevice.10.bc in the CUDA toolkit.
+/// Searches: $CUDA_TOOLKIT_PATH/nvvm/libdevice/libdevice.10.bc
+///           $CUDA_TOOLKIT_PATH/../nvvm/libdevice/libdevice.10.bc
+///           Common NixOS paths
+fn find_libdevice(config: &PipelineConfig) -> Result<PathBuf> {
+    let candidates = [
+        config.cuda_toolkit_path.join("nvvm/libdevice/libdevice.10.bc"),
+        config.cuda_toolkit_path.join("../cuda_nvcc/nvvm/libdevice/libdevice.10.bc"),
+    ];
+    
+    // Also check CUDA_OXIDE_LIBDEVICE env var
+    if let Ok(path) = std::env::var("CUDA_OXIDE_LIBDEVICE") {
+        let p = PathBuf::from(path);
+        if p.exists() { return Ok(p); }
+    }
+    
+    for p in &candidates {
+        if p.exists() { return Ok(p.clone()); }
+    }
+    
+    Err(anyhow::anyhow!(
+        "libdevice.10.bc not found. Set CUDA_OXIDE_LIBDEVICE=/path/to/libdevice.10.bc"
+    ))
+}
+
+/// Find an LLVM tool (llvm-link, opt) next to the configured llc binary.
+fn find_llvm_tool(tool: &str, config: &PipelineConfig) -> Result<PathBuf> {
+    let llc_dir = config.llc_path.parent().unwrap_or(Path::new("/usr/bin"));
+    let tool_path = llc_dir.join(tool);
+    if tool_path.exists() {
+        Ok(tool_path)
+    } else {
+        // Fall back to PATH
+        Ok(PathBuf::from(tool))
+    }
+}
