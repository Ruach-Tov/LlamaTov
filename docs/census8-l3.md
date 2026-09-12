# Census 8 — KernelBench Level 3 Store-Unit Status

**Census:** 8 (Mavdil, September 12, 2026)
**Result:** 20 of 23 store-units BIT_EXACT, of 50 total L3 problems

**Reference environment:**
- Oracle: PyTorch 2.7.0 CPU (`stock_torch_cpu_default`)
- Device: Tesla P4 (sm_61, Pascal)
- CUDA: 12.8
- `fma_mode`: default (no `--fmad=false`)

## Per-problem status (23 attempted)

### BIT_EXACT (20)

| # | Problem | Segments | Status | Notes |
|---|---|---:|---|---|
| 1 | SimpleNet conv→relu→pool→fc | 5 | BIT_EXACT | baseline |
| 2 | SimpleReLU | 1 | BIT_EXACT | trivial |
| 3 | SimpleConv chain | 3 | BIT_EXACT | |
| 4 | Conv→BN→ReLU | 3 | BIT_EXACT | |
| 5 | AlexNet-like | 8 | BIT_EXACT | |
| 6 | GoogleNet inception | 12 | BIT_EXACT | multi-branch |
| 7 | VGG-like deep | 16 | BIT_EXACT | |
| 8 | ResNet-like skip | 10 | BIT_EXACT | skip connections |
| 9 | DenseNet-like | 14 | BIT_EXACT | dense connections |
| 10 | Inception-v3-like | 18 | BIT_EXACT | |
| 11 | SqueezeNet-like | 8 | BIT_EXACT | fire modules |
| 14 | UNet-like | 20 | BIT_EXACT | encoder-decoder |
| 15 | ResNeXt-like | 12 | BIT_EXACT | grouped conv |
| 16 | ShuffleNet-like | 10 | BIT_EXACT | channel shuffle |
| 17 | SqueezeExcite | 8 | BIT_EXACT | attention |
| 18 | GhostNet-like | 10 | BIT_EXACT | ghost modules |
| 19 | MobileNetV1 | 27 | BIT_EXACT | depthwise-separable |
| 20 | MobileNetV2 | 35 | BIT_EXACT | inverted residuals |
| 21 | EfficientNet MBConv | 2 islands | BIT_EXACT | container-reach |
| 22 | RegNet-like | 12 | BIT_EXACT | |

### NOT YET PASSING (3)

| # | Problem | Status | Reason |
|---|---|---|---|
| 12 | WideResNet-like | DIFF | branch-cat reconstruction gap |
| 13 | Swin-like transformer | DIFF | method-call emission (window attention) |
| 23 | EfficientNetV2-like | DIFF | fused-MBConv variant not yet emitted |

### NOT YET ATTEMPTED (27)

Problems #24-50, including:
- **#36-42**: cuDNN RNN family (LSTM/GRU). Pass the benchmark's own
  tolerance (our residual 2.4e-7 vs the benchmark's 1e-2 bar — four
  orders inside) but are not bit-exact. Lifting the recurrence to
  declarative form is in progress.
- **#24-35, #43-50**: not yet in the emission pipeline.

## Container-reach family (the linker proof)

The three container-reach problems (#19, #20, #21) prove the linker
can resolve symbols across multi-flow computation graphs:

- **#19 MobileNetV1** (27 segments): depthwise-separable pipeline
- **#20 MobileNetV2** (35 segments): inverted residual with expansion/projection
- **#21 EfficientNet** (2 computation islands): island-rebinding across the split

All three gate at zero (0 diff elements across all segments).

---

*Ruach Tov Collective, September 12, 2026.*
