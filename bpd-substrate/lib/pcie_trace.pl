%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
%% pcie_trace.pl — Prolog-callable PCIe and GPU tracing
%% 
%% Usage from Scheme dispatch:
%%   (dispatch-mcp "scheme_eval" {"program": "(pcie-trace-gpu \"relu\" 1000000)"})
%%
%% Sweepable parameters exposed:
%%   pcie_gen(Gen)       — current PCIe generation
%%   pcie_width(Width)   — current lane width  
%%   pcie_max_gen(Gen)   — max supported generation
%%   pcie_max_width(W)   — max supported width
%%   pcie_bandwidth_theoretical(BW_MBps) — theoretical max
%%   pcie_bandwidth_measured(BW_MBps)    — actual measured

:- module(pcie_trace, [
    pcie_config/4,
    pcie_bandwidth/2,
    gpu_transfer_rate/3,
    pcie_bottleneck/1
]).

%% pcie_config(Gen, Width, MaxGen, MaxWidth)
%% Query via: nvidia-smi --query-gpu=pcie.link.gen.current,...
pcie_config(Gen, Width, MaxGen, MaxWidth) :-
    % These would be populated by a shell call
    Gen = 1, Width = 8, MaxGen = 3, MaxWidth = 16.

%% Theoretical PCIe bandwidth in MB/s per direction
%% Gen1: 250 MB/s/lane, Gen2: 500, Gen3: ~1000, Gen4: ~2000, Gen5: ~4000
pcie_bandwidth(Gen, Width, BW_MBps) :-
    gen_rate(Gen, Rate),
    BW_MBps is Rate * Width.

gen_rate(1, 250).
gen_rate(2, 500).
gen_rate(3, 985).   % ~1 GB/s per lane after encoding overhead
gen_rate(4, 1969).
gen_rate(5, 3938).

%% Detect PCIe bottleneck
pcie_bottleneck(Reason) :-
    pcie_config(Gen, Width, MaxGen, MaxWidth),
    (Gen < MaxGen -> Reason = gen_downgrade(Gen, MaxGen)
    ; Width < MaxWidth -> Reason = width_downgrade(Width, MaxWidth)
    ; Reason = none).
