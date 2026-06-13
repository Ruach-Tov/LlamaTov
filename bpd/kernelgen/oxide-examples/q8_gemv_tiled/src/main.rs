#![allow(clippy::needless_range_loop)]
// SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
// Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_device::shared::DynamicSharedArray;
use cuda_host::cuda_module;
use std::time::Instant;

// tiled_v4 port: BM warps/block, each warp = 1 row; Xq(i32 words)+Xd(f32) STAGED in shared mem,
// loaded once per block, reused across BM rows. i32-word loads for Wq/Xq. Mirrors the CUDA kernel.
#[cuda_module]
mod kernels {
    use super::*;
    #[kernel]
    pub fn q8_gemv(wq: &[i32], wd: &[f32], xq: &[i32], xd: &[f32], mut y: DisjointSlice<f32>, k: u32, m: u32) {
        const BM: usize = 16;
        let tid = thread::threadIdx_x() as usize;
        let bdim = thread::blockDim_x() as usize;
        let warp = tid >> 5;
        let lane = tid & 31;
        let row = (thread::blockIdx_x() as usize) * BM + warp;
        let k_words = (k as usize) / 4;     // i32 words in X
        let nb = (k as usize) / 32;
        // shared: sXq (k_words i32) then sXd (nb f32)
        let s_xq: *mut i32 = DynamicSharedArray::<i32>::get();
        let s_xd: *mut f32 = DynamicSharedArray::<f32>::offset(k_words * 4);
        // cooperative load of X into shared
        let mut i = tid;
        while i < k_words { unsafe { *s_xq.add(i) = *xq.get_unchecked(i); } i += bdim; }
        let mut i2 = tid;
        while i2 < nb { unsafe { *s_xd.add(i2) = *xd.get_unchecked(i2); } i2 += bdim; }
        thread::sync_threads();
        if row >= m as usize { return; }
        let dot4 = |w: i32, x: i32| -> i32 {
            let w0=(w as i8) as i32; let x0=(x as i8) as i32;
            let w1=((w>>8) as i8) as i32; let x1=((x>>8) as i8) as i32;
            let w2=((w>>16) as i8) as i32; let x2=((x>>16) as i8) as i32;
            let w3=((w>>24) as i8) as i32; let x3=((x>>24) as i8) as i32;
            w0*x0 + w1*x1 + w2*x2 + w3*x3
        };
        let mut acc = 0.0f32;
        let mut b = lane;
        while b < nb {
            let wbase = row * k_words + b * 8;
            let xbase = b * 8;
            let mut isum: i32 = 0;
            let mut jw = 0usize;
            while jw < 8 {
                let ww = unsafe { *wq.get_unchecked(wbase + jw) };
                let xw = unsafe { *s_xq.add(xbase + jw) };   // X from SHARED
                isum += dot4(ww, xw);
                jw += 1;
            }
            let wdv = unsafe { *wd.get_unchecked(row * nb + b) };
            let xdv = unsafe { *s_xd.add(b) };               // Xd from SHARED
            acc = (wdv * xdv).mul_add(isum as f32, acc);
            b += 32;
        }
        acc += warp::shuffle_down_f32(acc, 16);
        acc += warp::shuffle_down_f32(acc, 8);
        acc += warp::shuffle_down_f32(acc, 4);
        acc += warp::shuffle_down_f32(acc, 2);
        acc += warp::shuffle_down_f32(acc, 1);
        if lane == 0 { unsafe { *y.get_unchecked_mut(row) = acc; } }
    }
}

const M: usize = 256;
const K: usize = 896;
const ITERS: usize = 2000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== q8_0 GEMV (RCo tiled_v4: shared-X staging, BM=16) — check + bench ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module_from_file("q8_gemv.ptx")?;
    let module = kernels::from_module(module)?;
    let nb = K/32; let kw = K/4;
    let smem_bytes = (kw*4 + nb*4) as u32;   // sXq (i32) + sXd (f32)
    let pack = |v:&[i8]| -> Vec<i32> { v.chunks(4).map(|c| i32::from_le_bytes([c[0] as u8,c[1] as u8,c[2] as u8,c[3] as u8])).collect() };
    // --- correctness ---
    {
        let mut wq8=vec![0i8;M*K]; let mut xq8=vec![0i8;K];
        for i in 0..M*K { wq8[i]=(((i*31+7)%255) as i32 -127) as i8; }
        for i in 0..K { xq8[i]=(((i*17+5)%255) as i32 -127) as i8; }
        let wq=pack(&wq8); let xq=pack(&xq8);
        let mut wd=vec![0f32;M*nb]; let mut xd=vec![0f32;nb];
        for i in 0..M*nb { wd[i]=half::f16::from_f32((((i*13+3)%100) as f32)*0.0007).to_f32(); }
        for i in 0..nb { xd[i]=half::f16::from_f32((((i*11+2)%100) as f32)*0.0009).to_f32(); }
        let mut cpu=vec![0f32;M];
        for row in 0..M { let mut la=[0f32;32];
            for lane in 0..32 { let mut acc=0.0f32; let mut b=lane;
                while b<nb { let mut isum=0i32; for j in 0..32 { isum+=(wq8[row*K+b*32+j] as i32)*(xq8[b*32+j] as i32); }
                    acc=(wd[row*nb+b]*xd[b]).mul_add(isum as f32,acc); b+=32; } la[lane]=acc; }
            for s in [16usize,8,4,2,1] { for t in 0..s { la[t]+=la[t+s]; } } cpu[row]=la[0]; }
        let wq_d=DeviceBuffer::from_host(&stream,&wq)?; let wd_d=DeviceBuffer::from_host(&stream,&wd)?;
        let xq_d=DeviceBuffer::from_host(&stream,&xq)?; let xd_d=DeviceBuffer::from_host(&stream,&xd)?;
        let mut y_d=DeviceBuffer::from_host(&stream,&vec![0f32;M])?;
        let cfg=LaunchConfig{grid_dim:((M as u32).div_ceil(16),1,1),block_dim:(16*32,1,1),shared_mem_bytes:smem_bytes};
        module.q8_gemv(stream.as_ref(),cfg,&wq_d,&wd_d,&xq_d,&xd_d,&mut y_d,K as u32,M as u32)?;
        stream.synchronize()?;
        let y=y_d.to_host_vec(&stream)?;
        let mut mu=0i64; for i in 0..M { let u=((cpu[i].to_bits() as i64)-(y[i].to_bits() as i64)).abs(); if u>mu{mu=u;} }
        println!("  CORRECTNESS: max_ulp={} ({})", mu, if mu==0 {"0-ULP OK"} else {"DIFFERS"});
    }
    // --- bench ---
    for (m,label) in [(896usize,"o_proj"),(4864,"down_proj"),(151936,"lm_head")] {
        let wq=vec![0x01010101i32; m*kw]; let wd=vec![0.01f32; m*nb];
        let xq=vec![0x01010101i32; kw]; let xd=vec![0.01f32; nb];
        let wq_d=DeviceBuffer::from_host(&stream,&wq)?; let wd_d=DeviceBuffer::from_host(&stream,&wd)?;
        let xq_d=DeviceBuffer::from_host(&stream,&xq)?; let xd_d=DeviceBuffer::from_host(&stream,&xd)?;
        let mut y_d=DeviceBuffer::from_host(&stream,&vec![0f32;m])?;
        let cfg=LaunchConfig{grid_dim:((m as u32).div_ceil(16),1,1),block_dim:(16*32,1,1),shared_mem_bytes:smem_bytes};
        for _ in 0..50 { module.q8_gemv(stream.as_ref(),cfg,&wq_d,&wd_d,&xq_d,&xd_d,&mut y_d,K as u32,m as u32)?; }
        stream.synchronize()?;
        let t0=Instant::now();
        for _ in 0..ITERS { module.q8_gemv(stream.as_ref(),cfg,&wq_d,&wd_d,&xq_d,&xd_d,&mut y_d,K as u32,m as u32)?; }
        stream.synchronize()?;
        let us=t0.elapsed().as_secs_f64()*1e6/ITERS as f64;
        let wbytes=(m*K+m*nb*2) as f64;
        println!("  {:10} M={:6} : {:8.2} us/call   {:6.1} GB/s", label, m, us, wbytes/(us*1e3));
    }
    Ok(())
}
