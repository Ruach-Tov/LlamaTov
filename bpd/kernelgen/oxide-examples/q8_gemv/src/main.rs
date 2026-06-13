// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
/*
 * q8_0 dp4a GEMV — fact-lowered to Rust/cuda-oxide (the thesis: same facts, different backend).
 * Mirrors the CUDA emitter's tiled_v4 q8_0 GEMV EXACTLY:
 *   reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_xor,5))
 * One WARP per output row. Lane b processes blocks b, b+32, ...; per block an exact int8 dp4a
 * (sum of 32 int8*int8 products as i32 — bit-identical to __dp4a, integer/no rounding), scaled by
 * wd*xd and fma-accumulated; then a 5-step shuffle_xor_f32 butterfly warp reduction; lane 0 writes.
 * Build: cargo oxide run q8_gemv --arch sm_61
 */
#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// q8_0 GEMV (canonical_serial) — reproduces the CUDA emitter's emit_q8_0_gemv_canonical_serial
    /// reduction order EXACTLY, so it is bit-identical to the CUDA kernel (the true cross-backend
    /// claim). reduction_order(q8_gemv_dp4a, lanes(32), strided, accum(fma), tree(shfl_down,5)):
    ///   1) 32 lane-partials: lane L folds blocks b=L,L+32,... with FUSED multiply-add (mul_add)
    ///   2) 5-level shuffle-down tree merge (s=16,8,4,2,1): lane_acc[t] += lane_acc[t+s]
    /// ONE THREAD per row (matches the CUDA kernel, which also does the 32-partials + tree in one
    /// thread via a float lane_acc[32]) — so NO warp shuffle needed, sidestepping the sm_61
    /// .sync-shuffle lowering block. dp4a = exact integer dot (== __dp4a). Scales: f32 (= __half2float
    /// of the q8_0 fp16 scales, computed host-side identically). m via y.len(), k via arg.
    #[kernel]
    pub fn q8_gemv(
        wq: &[i8],
        wd: &[f32],
        xq: &[i8],
        xd: &[f32],
        mut y: DisjointSlice<f32>,
        k: u32,
    ) {
        let gid = thread::index_1d();
        let row = gid.get();
        let m = y.len();
        if row >= m {
            return;
        }
        let k_size = k as usize;
        let nb = k_size / 32;
        let mut lane_acc = [0.0f32; 32];
        // 1) 32 lane-partials, each an fma-contracted strided fold (matches __fmaf_rn)
        let mut lane = 0usize;
        while lane < 32 {
            let mut acc = 0.0f32;
            let mut b = lane;
            while b < nb {
                let mut isum: i32 = 0;
                let base = b * 32;
                let mut j = 0usize;
                while j < 32 {
                    isum += (wq[row * k_size + base + j] as i32) * (xq[base + j] as i32);
                    j += 1;
                }
                let wdv = wd[row * nb + b];
                let xdv = xd[b];
                acc = (wdv * xdv).mul_add(isum as f32, acc); // FUSED: matches __fmaf_rn(wd*xd, isum, acc)
                b += 32;
            }
            lane_acc[lane] = acc;
            lane += 1;
        }
        // 2) 5-level shuffle-down tree merge (s=16,8,4,2,1) — same as the warp reduction
        let mut s = 16usize;
        while s > 0 {
            let mut t = 0usize;
            while t < s {
                lane_acc[t] += lane_acc[t + s];
                t += 1;
            }
            s >>= 1;
        }
        unsafe {
            *y.get_unchecked_mut(row) = lane_acc[0];
        }
    }
}

// =========================== HOST ===========================
const M: usize = 256;
const K: usize = 896; // nb = 28

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== q8_0 dp4a GEMV via cuda-oxide (fact-lowered Rust kernel) ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let nb = K / 32;

    let mut wq = vec![0i8; M * K];
    let mut wd = vec![0f32; M * nb];
    let mut xq = vec![0i8; K];
    let mut xd = vec![0f32; nb];
    for i in 0..M * K {
        wq[i] = (((i * 31 + 7) % 255) as i32 - 127) as i8;
    }
    // scales generated as fp16 then converted to f32 (== __half2float), so the f32 oxide sees is
    // EXACTLY what the CUDA kernel computes from the same fp16 — making the cross-check bit-valid.
    let mut wd_h = vec![half::f16::ZERO; M * nb];
    let mut xd_h = vec![half::f16::ZERO; nb];
    for i in 0..M * nb {
        wd_h[i] = half::f16::from_f32((((i * 13 + 3) % 100) as f32) * 0.0007);
        wd[i] = wd_h[i].to_f32();
    }
    for i in 0..K {
        xq[i] = (((i * 17 + 5) % 255) as i32 - 127) as i8;
    }
    for i in 0..nb {
        xd_h[i] = half::f16::from_f32((((i * 11 + 2) % 100) as f32) * 0.0009);
        xd[i] = xd_h[i].to_f32();
    }

    let cpu = q8_gemv_reference(&wq, &wd, &xq, &xd, M, K);

    let wq_dev = DeviceBuffer::from_host(&stream, &wq)?;
    let wd_dev = DeviceBuffer::from_host(&stream, &wd)?;
    let xq_dev = DeviceBuffer::from_host(&stream, &xq)?;
    let xd_dev = DeviceBuffer::from_host(&stream, &xd)?;
    let mut y_dev = DeviceBuffer::from_host(&stream, &vec![0f32; M])?;

    let module = ctx.load_module_from_file("q8_gemv.ptx")?;
    let module = kernels::from_module(module)?;

    // one thread per row (row = global thread id)
    let block_x = 128u32;
    let grid_x = (M as u32).div_ceil(block_x);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_x, 1, 1),
        shared_mem_bytes: 0,
    };
    module.q8_gemv(
        stream.as_ref(), cfg,
        &wq_dev, &wd_dev, &xq_dev, &xd_dev, &mut y_dev, K as u32,
    )?;
    stream.synchronize()?;
    let y = y_dev.to_host_vec(&stream)?;

    let mut max_ulp = 0i64;
    let mut max_abs = 0f32;
    for i in 0..M {
        let u = ((cpu[i].to_bits() as i64) - (y[i].to_bits() as i64)).abs();
        if u > max_ulp { max_ulp = u; }
        let d = (cpu[i] - y[i]).abs();
        if d > max_abs { max_abs = d; }
    }
    println!("  M={} K={} nb={}", M, K, nb);
    println!("  oxide y[0..3] = {:?}", &y[0..3]);
    println!("  ref   y[0..3] = {:?}", &cpu[0..3]);
    println!("  max_ulp = {}  max_abs = {:e}", max_ulp, max_abs);
    if max_ulp == 0 {
        println!("  \u{2713} 0-ULP: oxide q8_0 GEMV BIT-IDENTICAL to reference");
    } else {
        println!("  \u{2717} DIFFERS by {} ULP", max_ulp);
    }

    // dump inputs + oxide output for the cross-backend gate (CUDA emitter runs the SAME inputs).
    // Raw little-endian: header [M,K] as i32, then wq(i8 M*K), wd(f32 M*nb), xq(i8 K), xd(f32 nb),
    // y(f32 M). The Python harness reads this, runs the CUDA canonical_serial kernel, compares.
    if std::env::var("Q8_DUMP").is_ok() {
        use std::io::Write;
        let path = std::env::var("Q8_DUMP").unwrap();
        let mut f = std::fs::File::create(&path)?;
        // header [M,K] i32; wq i8 M*K; wd_h fp16 M*nb; xq i8 K; xd_h fp16 nb; y f32 M.
        // Scales dumped as FP16 (the CUDA kernel reads __half) — oxide's f32 == __half2float(these).
        f.write_all(&(M as i32).to_le_bytes())?;
        f.write_all(&(K as i32).to_le_bytes())?;
        f.write_all(unsafe { std::slice::from_raw_parts(wq.as_ptr() as *const u8, wq.len()) })?;
        for v in &wd_h { f.write_all(&v.to_bits().to_le_bytes())?; }
        f.write_all(unsafe { std::slice::from_raw_parts(xq.as_ptr() as *const u8, xq.len()) })?;
        for v in &xd_h { f.write_all(&v.to_bits().to_le_bytes())?; }
        for v in &y { f.write_all(&v.to_le_bytes())?; }
        println!("  dumped inputs(fp16 scales)+output -> {}", path);
    }
    Ok(())
}

/// CPU reference replicating the canonical_serial reduction order EXACTLY (32 fma-folded lane
/// partials over strided blocks, then the 5-level shuffle-DOWN tree merge). Matches both the oxide
/// kernel and the CUDA emitter's emit_q8_0_gemv_canonical_serial.
fn q8_gemv_reference(wq: &[i8], wd: &[f32], xq: &[i8], xd: &[f32], m: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut out = vec![0f32; m];
    for row in 0..m {
        let mut lane_acc = [0f32; 32];
        for lane in 0..32 {
            let mut acc = 0.0f32;
            let mut b = lane;
            while b < nb {
                let mut isum: i32 = 0;
                let base = b * 32;
                for j in 0..32 {
                    isum += (wq[row * k + base + j] as i32) * (xq[base + j] as i32);
                }
                acc = (wd[row * nb + b] * xd[b]).mul_add(isum as f32, acc); // fused
                b += 32;
            }
            lane_acc[lane] = acc;
        }
        let mut s = 16usize;
        while s > 0 {
            for t in 0..s {
                lane_acc[t] += lane_acc[t + s];
            }
            s >>= 1;
        }
        out[row] = lane_acc[0];
    }
    out
}
