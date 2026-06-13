#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_device::shared::DynamicSharedArray;
use cuda_host::cuda_module;

// rms->quant seam (oxide lowering of activation_fold(rms_norm)). Two-phase, both canonical orders
// preserved -> bit-identical to CUDA k_rms_quant. PHASE1: thread 0 serial sum-of-squares -> inv
// (shared broadcast). PHASE2: per 32-block warp-amax quantize of nv = x*inv*nw.
// device-side f32<->f16 round-to-nearest-even, bit-identical to __float2half/__half2float.
// (the `half` crate uses x86 intrinsics that don't lower to PTX, so we do it in pure arithmetic.)
#[inline(always)]
fn f32_to_f16_bits(f: f32) -> u32 {
    let x = f.to_bits();
    let sign = (x >> 16) & 0x8000;
    let mut mant = x & 0x007fffff;
    let exp = ((x >> 23) & 0xff) as i32;
    if exp == 0xff {
        // inf/nan
        return sign | 0x7c00 | (if mant != 0 { 0x200 } else { 0 });
    }
    let mut e = exp - 127 + 15;
    if e >= 0x1f { return sign | 0x7c00; }          // overflow -> inf
    if e <= 0 {
        if e < -10 { return sign; }                 // underflow -> 0
        mant |= 0x00800000;
        let shift = (14 - e) as u32;
        let r = (mant >> (shift - 1)) & 1;
        let s = (mant & ((1u32 << (shift - 1)) - 1)) != 0;
        let l = (mant >> shift) & 1;
        let mut m16 = mant >> shift;
        if r == 1 && (s || l == 1) { m16 += 1; }
        return sign | (m16 & 0x7ff);
    }
    // normalized
    let r = (mant >> 12) & 1;
    let s = (mant & 0xfff) != 0;
    let l = (mant >> 13) & 1;
    let mut m16 = mant >> 13;
    let mut ee = e as u32;
    if r == 1 && (s || l == 1) {
        m16 += 1;
        if m16 == 0x400 { m16 = 0; ee += 1; if ee >= 0x1f { return sign | 0x7c00; } }
    }
    sign | (ee << 10) | (m16 & 0x3ff)
}
#[inline(always)]
fn f16_bits_to_f32(h: u32) -> f32 {
    let sign = (h & 0x8000) << 16;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        // subnormal
        let mut e = -1i32; let mut m = mant;
        while (m & 0x400) == 0 { m <<= 1; e -= 1; }
        m &= 0x3ff;
        let fe = (e + 1 - 15 + 127) as u32;
        return f32::from_bits(sign | (fe << 23) | (m << 13));
    }
    if exp == 0x1f { return f32::from_bits(sign | 0x7f800000 | (mant << 13)); }
    let fe = (exp as i32 - 15 + 127) as u32;
    f32::from_bits(sign | (fe << 23) | (mant << 13))
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn k_rms_quant(x: &[f32], nw: &[f32], mut xq: DisjointSlice<i32>, mut xd: DisjointSlice<f32>, k: u32, eps: f32) {
        let tid = thread::threadIdx_x() as usize;
        let nth = thread::blockDim_x() as usize;
        let ksz = k as usize;
        let nb = ksz / 32;
        // PHASE 1: block_row reduction order (strided 256-thread partials + shared-mem pairwise
        // tree), reproducing reduction_order(rms_ss, lanes(256), strided, tree(pairwise,8)) EXACTLY
        // -- the SAME order as the production block_row k_rmsnorm (and the CUDA k_rms_quant). This is
        // the correctness contract: NOT a serial left-fold. shared: nth f32 for the tree.
        let sred: *mut f32 = DynamicSharedArray::<f32>::get();
        let mut local = 0.0f32;
        let mut j = tid;
        while j < ksz { let v = unsafe { *x.get_unchecked(j) }; local += v*v; j += nth; }
        unsafe { *sred.add(tid) = local; }
        thread::sync_threads();
        let mut s = nth >> 1;
        while s > 0 {
            if tid < s { unsafe { *sred.add(tid) += *sred.add(tid + s); } }
            thread::sync_threads();
            s >>= 1;
        }
        let inv = (unsafe { *sred } / (ksz as f32) + eps).sqrt().recip();  // rsqrt(mean+eps)
        thread::sync_threads();
        // PHASE 2: per 32-block warp-amax quantize (matches standalone k_quant_q8 order)
        let warps = nth >> 5;
        let lane = tid & 31;
        let mut b = tid >> 5;
        while b < nb {
            let idx = b*32 + lane;
            let xv = unsafe { *x.get_unchecked(idx) };
            let nwv = unsafe { *nw.get_unchecked(idx) };
            let nv = xv * inv * nwv;
            let mut a = nv.abs();
            // warp-amax (shfl_down, max)
            let o = warp::shuffle_down_f32(a, 16); if o > a { a = o; }
            let o = warp::shuffle_down_f32(a, 8);  if o > a { a = o; }
            let o = warp::shuffle_down_f32(a, 4);  if o > a { a = o; }
            let o = warp::shuffle_down_f32(a, 2);  if o > a { a = o; }
            let o = warp::shuffle_down_f32(a, 1);  if o > a { a = o; }
            let amax = warp::shuffle_f32(a, 0);    // broadcast lane 0's amax
            let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let dh_bits = f32_to_f16_bits(d);
            if lane == 0 { unsafe { *xd.get_unchecked_mut(b) = dh_bits as f32; } }
            let dq = f16_bits_to_f32(dh_bits);
            let mut q = (nv / dq).round() as i32;
            if q < -127 { q = -127; } if q > 127 { q = 127; }
            // pack into i32 output (one int8 per i32 slot for simplicity of the check)
            unsafe { *xq.get_unchecked_mut(idx) = q; }
            b += warps;
        }
    }
}

const K: usize = 896;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== oxide k_rms_quant (rms->quant seam) self-check ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("rms_quant_oxide.ptx")?;
    let module = kernels::from_module(module)?;
    let nb = K/32; let eps = 1e-5f32;
    // deterministic input (same generator the CUDA xcheck can mirror)
    let mut x = vec![0f32; K]; let mut nw = vec![0f32; K];
    for i in 0..K { x[i] = (((i*37+11)%200) as f32 - 100.0) * 0.03; nw[i] = (((i*19+5)%100) as f32)*0.01 + 0.5; }
    // CPU reference: block_row rms (strided 256-lane partials + pairwise tree, == the kernel) then
    // canonical block-amax quantize. Reproduces reduction_order(rms_ss, lanes(256), strided, tree).
    const NTH: usize = 256;
    let mut sred = vec![0f32; NTH];
    for t in 0..NTH { let mut acc = 0.0f32; let mut j = t; while j < K { acc += x[j]*x[j]; j += NTH; } sred[t] = acc; }
    let mut s = NTH >> 1;
    while s > 0 { for t in 0..s { sred[t] += sred[t+s]; } s >>= 1; }
    let inv = (sred[0]/(K as f32)+eps).sqrt().recip();
    let mut xq_ref = vec![0i32; K]; let mut xd_ref = vec![0u16; nb];
    for b in 0..nb {
        let mut amax = 0.0f32;
        for l in 0..32 { let nv = x[b*32+l]*inv*nw[b*32+l]; if nv.abs() > amax { amax = nv.abs(); } }
        let d = if amax > 0.0 { amax/127.0 } else { 1.0 };
        let dh = half::f16::from_f32(d); xd_ref[b] = dh.to_bits();
        let dq = dh.to_f32();
        for l in 0..32 { let nv = x[b*32+l]*inv*nw[b*32+l]; let mut q=(nv/dq).round() as i32; if q < -127 { q = -127; } if q > 127 { q = 127; } xq_ref[b*32+l]=q; }
    }
    let x_d=DeviceBuffer::from_host(&stream,&x)?; let nw_d=DeviceBuffer::from_host(&stream,&nw)?;
    let mut xq_d=DeviceBuffer::from_host(&stream,&vec![0i32;K])?;
    let mut xd_d=DeviceBuffer::from_host(&stream,&vec![0f32;nb])?;
    let cfg=LaunchConfig{grid_dim:(1,1,1),block_dim:(256,1,1),shared_mem_bytes:256*4};  // block_row tree
    module.k_rms_quant(stream.as_ref(),cfg,&x_d,&nw_d,&mut xq_d,&mut xd_d,K as u32,eps)?;
    stream.synchronize()?;
    let xq=xq_d.to_host_vec(&stream)?; let xd=xd_d.to_host_vec(&stream)?;
    let xq_diff=(0..K).filter(|&i| xq[i]!=xq_ref[i]).count();
    let xd_diff=(0..nb).filter(|&b| (xd[b] as u32 as u16)!=xd_ref[b]).count();
    println!("  Xq differing: {}/{}  Xd differing: {}/{}", xq_diff, K, xd_diff, nb);
    println!("  Xq oxide[:8]={:?}", &xq[..8]);
    if xq_diff==0 && xd_diff==0 { println!("  oxide k_rms_quant 0-ULP vs canonical reference"); }
    else { println!("  DIFFERS"); }
    // dump inputs (x,nw f32) + oxide outputs (xq as i8, xd as f16 bits) for the cross-backend gate
    if let Ok(path) = std::env::var("RQ_DUMP") {
        use std::io::Write;
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&(K as i32).to_le_bytes())?;
        for v in &x { f.write_all(&v.to_le_bytes())?; }
        for v in &nw { f.write_all(&v.to_le_bytes())?; }
        for v in &xq { f.write_all(&[(*v as i8) as u8])?; }              // K int8
        for v in &xd { f.write_all(&((*v as u32) as u16).to_le_bytes())?; } // nb f16 bits
        println!("  dumped inputs+oxide output -> {}", path);
    }
    Ok(())
}
