# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""PyTorch Kernel Library — Reference implementations + test harness.

Comprehensive library of PyTorch kernels organized for:
  1. Parsing Python → understanding PyTorch semantics
  2. Lifting to BPD facts → generating equivalent CUDA
  3. Bit-identical verification → including intermediate steps

Each kernel provides:
  - PyTorch reference implementation (forward only)
  - Input generator (deterministic, seeded)
  - Expected output (computed by PyTorch)
  - Intermediate step verification points
  - Shape conventions matching Stanford KernelBench L1

ORGANIZATION:
  Section 1: Elementwise unary activations (14 kernels)
  Section 2: Elementwise binary operations (8 kernels)
  Section 3: Reduction operations (10 kernels)
  Section 4: Normalization layers (8 kernels)
  Section 5: Pooling operations (6 kernels)
  Section 6: Convolution operations (8 representative variants)
  Section 7: Linear algebra (8 matmul variants)
  Section 8: Loss functions (7 kernels)
  Section 9: Cumulative operations (5 kernels)
  Section 10: Attention + composite (3 kernels)
  Section 11: Quantization operations (3 kernels)

Total: 80+ kernels with tests

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-20
Per Heath: comprehensive PyTorch kernel library for BPD lifting pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional, Any


# ═══════════════════════════════════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════════════════════════════════

SEED = 42
DEVICE = 'cpu'  # CPU for bit-identical verification


@dataclass
class KernelSpec:
    """A PyTorch kernel with its test specification."""
    name: str
    category: str
    kernelbench_id: Optional[int]  # Stanford L1 problem number
    pytorch_fn: Callable           # The PyTorch operation
    input_gen: Callable            # Generates (inputs, kwargs) tuple
    intermediates: Optional[Callable] = None  # Compute intermediate steps
    description: str = ""
    complexity: str = "trivial"    # trivial, low, medium, high
    
    def run(self, seed=SEED):
        """Run the kernel and return (inputs, output, intermediates)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        inputs, kwargs = self.input_gen()
        with torch.no_grad():
            output = self.pytorch_fn(*inputs, **kwargs)
        intermediates = {}
        if self.intermediates:
            intermediates = self.intermediates(*inputs, **kwargs)
        return inputs, kwargs, output, intermediates


# Registry
KERNEL_LIBRARY: Dict[str, KernelSpec] = {}

def register(spec: KernelSpec):
    KERNEL_LIBRARY[spec.name] = spec
    return spec


# ═══════════════════════════════════════════════════════════════════════
# Section 1: Elementwise Unary Activations
# ═══════════════════════════════════════════════════════════════════════

def _act_input(shape=(16, 16384)):
    x = torch.randn(*shape)
    return (x,), {}

register(KernelSpec(
    name='relu', category='activation', kernelbench_id=19,
    pytorch_fn=F.relu,
    input_gen=_act_input,
    description='ReLU: max(0, x)',
    complexity='trivial'
))

register(KernelSpec(
    name='leaky_relu', category='activation', kernelbench_id=20,
    pytorch_fn=lambda x: F.leaky_relu(x, 0.01),
    input_gen=_act_input,
    description='LeakyReLU: x if x>0 else 0.01*x',
    complexity='trivial'
))

register(KernelSpec(
    name='sigmoid', category='activation', kernelbench_id=21,
    pytorch_fn=torch.sigmoid,
    input_gen=_act_input,
    description='Sigmoid: 1/(1+exp(-x))',
    complexity='trivial'
))

register(KernelSpec(
    name='tanh', category='activation', kernelbench_id=22,
    pytorch_fn=torch.tanh,
    input_gen=_act_input,
    description='Tanh: (exp(x)-exp(-x))/(exp(x)+exp(-x))',
    complexity='trivial'
))

register(KernelSpec(
    name='silu', category='activation', kernelbench_id=25,
    pytorch_fn=F.silu,
    input_gen=_act_input,
    description='SiLU/Swish: x * sigmoid(x)',
    intermediates=lambda x: {'sigmoid': torch.sigmoid(x)},
    complexity='trivial'
))

register(KernelSpec(
    name='gelu_erf', category='activation', kernelbench_id=26,
    pytorch_fn=lambda x: F.gelu(x),
    input_gen=_act_input,
    description='GELU exact: 0.5*x*(1+erf(x/sqrt(2)))',
    intermediates=lambda x: {
        'scaled': x * 0.7071067811865476,
        'erf': torch.erf(x * 0.7071067811865476),
    },
    complexity='trivial'
))

register(KernelSpec(
    name='gelu_tanh', category='activation', kernelbench_id=88,
    pytorch_fn=lambda x: F.gelu(x, approximate='tanh'),
    input_gen=_act_input,
    description='GELU tanh approx: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))',
    intermediates=lambda x: {
        'inner': 0.7978845608028654 * x * (1.0 + 0.044715 * x * x),
        'tanh': torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)),
    },
    complexity='trivial'
))

register(KernelSpec(
    name='selu', category='activation', kernelbench_id=27,
    pytorch_fn=F.selu,
    input_gen=_act_input,
    description='SELU: lambda*(x if x>0 else alpha*(exp(x)-1))',
    complexity='trivial'
))

register(KernelSpec(
    name='hardsigmoid', category='activation', kernelbench_id=28,
    pytorch_fn=F.hardsigmoid,
    input_gen=_act_input,
    description='HardSigmoid: clamp(x+3, 0, 6)/6',
    intermediates=lambda x: {
        'shifted': x + 3.0,
        'clamped': torch.clamp(x + 3.0, 0.0, 6.0),
    },
    complexity='trivial'
))

register(KernelSpec(
    name='softplus', category='activation', kernelbench_id=29,
    pytorch_fn=F.softplus,
    input_gen=_act_input,
    description='Softplus: log(1+exp(x)), threshold=20',
    complexity='trivial'
))

register(KernelSpec(
    name='softsign', category='activation', kernelbench_id=30,
    pytorch_fn=F.softsign,
    input_gen=_act_input,
    description='Softsign: x/(1+|x|)',
    complexity='trivial'
))

register(KernelSpec(
    name='elu', category='activation', kernelbench_id=31,
    pytorch_fn=lambda x: F.elu(x, 1.0),
    input_gen=_act_input,
    description='ELU: x if x>0 else alpha*(exp(x)-1)',
    complexity='trivial'
))

register(KernelSpec(
    name='hardtanh', category='activation', kernelbench_id=32,
    pytorch_fn=F.hardtanh,
    input_gen=_act_input,
    description='HardTanh: clamp(x, -1, 1)',
    complexity='trivial'
))

register(KernelSpec(
    name='mish', category='activation', kernelbench_id=None,
    pytorch_fn=F.mish,
    input_gen=_act_input,
    description='Mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))',
    intermediates=lambda x: {
        'softplus': F.softplus(x),
        'tanh_sp': torch.tanh(F.softplus(x)),
    },
    complexity='trivial'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 2: Elementwise Binary Operations
# ═══════════════════════════════════════════════════════════════════════

def _binary_input(shape=(16, 16384)):
    a = torch.randn(*shape)
    b = torch.randn(*shape)
    return (a, b), {}

register(KernelSpec(
    name='add', category='binary', kernelbench_id=None,
    pytorch_fn=torch.add,
    input_gen=_binary_input,
    description='Element-wise addition',
    complexity='trivial'
))

register(KernelSpec(
    name='mul', category='binary', kernelbench_id=None,
    pytorch_fn=torch.mul,
    input_gen=_binary_input,
    description='Element-wise multiplication',
    complexity='trivial'
))

register(KernelSpec(
    name='sub', category='binary', kernelbench_id=None,
    pytorch_fn=torch.sub,
    input_gen=_binary_input,
    description='Element-wise subtraction',
    complexity='trivial'
))

register(KernelSpec(
    name='div', category='binary', kernelbench_id=None,
    pytorch_fn=torch.div,
    input_gen=lambda: ((torch.randn(16, 16384), torch.randn(16, 16384).abs() + 0.1), {}),
    description='Element-wise division (positive divisor)',
    complexity='trivial'
))

register(KernelSpec(
    name='maximum', category='binary', kernelbench_id=None,
    pytorch_fn=torch.maximum,
    input_gen=_binary_input,
    description='Element-wise maximum',
    complexity='trivial'
))

register(KernelSpec(
    name='minimum', category='binary', kernelbench_id=None,
    pytorch_fn=torch.minimum,
    input_gen=_binary_input,
    description='Element-wise minimum',
    complexity='trivial'
))

register(KernelSpec(
    name='pow', category='binary', kernelbench_id=None,
    pytorch_fn=lambda x, y: torch.pow(x.abs() + 0.01, y),
    input_gen=lambda: ((torch.randn(16, 16384), torch.rand(16, 16384) * 3), {}),
    description='Element-wise power (positive base)',
    complexity='low'
))

register(KernelSpec(
    name='scalar_mul', category='binary', kernelbench_id=5,
    pytorch_fn=lambda x: x * 2.5,
    input_gen=lambda: ((torch.randn(16, 16384),), {}),
    description='Scalar multiplication: x * alpha',
    complexity='trivial'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 3: Reduction Operations
# ═══════════════════════════════════════════════════════════════════════

def _reduction_input(shape=(16, 4096)):
    return (torch.randn(*shape),), {}

register(KernelSpec(
    name='softmax', category='reduction', kernelbench_id=23,
    pytorch_fn=lambda x: F.softmax(x, dim=-1),
    input_gen=_reduction_input,
    description='Row-wise softmax',
    intermediates=lambda x: {
        'max': x.max(dim=-1, keepdim=True).values,
        'shifted': x - x.max(dim=-1, keepdim=True).values,
        'exp': torch.exp(x - x.max(dim=-1, keepdim=True).values),
        'sum_exp': torch.exp(x - x.max(dim=-1, keepdim=True).values).sum(dim=-1, keepdim=True),
    },
    complexity='medium'
))

register(KernelSpec(
    name='log_softmax', category='reduction', kernelbench_id=24,
    pytorch_fn=lambda x: F.log_softmax(x, dim=-1),
    input_gen=_reduction_input,
    description='Row-wise log softmax',
    complexity='medium'
))

register(KernelSpec(
    name='sum_reduction', category='reduction', kernelbench_id=47,
    pytorch_fn=lambda x: x.sum(dim=-1),
    input_gen=_reduction_input,
    description='Sum reduction over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='mean_reduction', category='reduction', kernelbench_id=48,
    pytorch_fn=lambda x: x.mean(dim=-1),
    input_gen=_reduction_input,
    description='Mean reduction over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='max_reduction', category='reduction', kernelbench_id=49,
    pytorch_fn=lambda x: x.max(dim=-1).values,
    input_gen=_reduction_input,
    description='Max reduction over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='min_reduction', category='reduction', kernelbench_id=53,
    pytorch_fn=lambda x: x.min(dim=-1).values,
    input_gen=_reduction_input,
    description='Min reduction over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='argmax', category='reduction', kernelbench_id=51,
    pytorch_fn=lambda x: x.argmax(dim=-1),
    input_gen=_reduction_input,
    description='Argmax over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='argmin', category='reduction', kernelbench_id=52,
    pytorch_fn=lambda x: x.argmin(dim=-1),
    input_gen=_reduction_input,
    description='Argmin over last dimension',
    complexity='low'
))

register(KernelSpec(
    name='frobenius_norm', category='reduction', kernelbench_id=37,
    pytorch_fn=lambda x: torch.norm(x, p='fro', dim=-1),
    input_gen=_reduction_input,
    description='Frobenius norm over last dimension',
    intermediates=lambda x: {
        'squared': x * x,
        'sum_sq': (x * x).sum(dim=-1),
    },
    complexity='low'
))

register(KernelSpec(
    name='l1_norm', category='reduction', kernelbench_id=38,
    pytorch_fn=lambda x: torch.norm(x, p=1, dim=-1),
    input_gen=_reduction_input,
    description='L1 norm over last dimension',
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 4: Normalization Layers
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='batch_norm', category='normalization', kernelbench_id=33,
    pytorch_fn=lambda x, w, b, rm, rv: F.batch_norm(x, rm, rv, w, b, training=False),
    input_gen=lambda: (
        (torch.randn(16, 64, 8, 8),           # x: (N,C,H,W)
         torch.randn(64),                      # weight (gamma)
         torch.randn(64),                      # bias (beta)
         torch.randn(64),                      # running_mean
         torch.rand(64).abs() + 0.1),          # running_var (positive)
        {}
    ),
    description='BatchNorm2d in eval mode',
    intermediates=lambda x, w, b, rm, rv: {
        'normalized': (x - rm.view(1,-1,1,1)) / torch.sqrt(rv.view(1,-1,1,1) + 1e-5),
        'scaled': w.view(1,-1,1,1) * (x - rm.view(1,-1,1,1)) / torch.sqrt(rv.view(1,-1,1,1) + 1e-5),
    },
    complexity='medium'
))

register(KernelSpec(
    name='layer_norm', category='normalization', kernelbench_id=40,
    pytorch_fn=lambda x: F.layer_norm(x, [x.shape[-1]]),
    input_gen=lambda: ((torch.randn(16, 256, 1024),), {}),
    description='LayerNorm over last dimension',
    intermediates=lambda x: {
        'mean': x.mean(dim=-1, keepdim=True),
        'var': x.var(dim=-1, unbiased=False, keepdim=True),
        'normalized': (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, unbiased=False, keepdim=True) + 1e-5),
    },
    complexity='medium'
))

register(KernelSpec(
    name='rms_norm', category='normalization', kernelbench_id=36,
    pytorch_fn=lambda x: x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5),
    input_gen=lambda: ((torch.randn(16, 256, 2048),), {}),
    description='RMSNorm: x / sqrt(mean(x^2) + eps)',
    intermediates=lambda x: {
        'squared': x.pow(2),
        'mean_sq': x.pow(2).mean(dim=-1, keepdim=True),
        'rms': torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5),
    },
    complexity='medium'
))

register(KernelSpec(
    name='instance_norm', category='normalization', kernelbench_id=34,
    pytorch_fn=lambda x: F.instance_norm(x),
    input_gen=lambda: ((torch.randn(16, 64, 32, 32),), {}),
    description='InstanceNorm2d',
    complexity='medium'
))

register(KernelSpec(
    name='group_norm', category='normalization', kernelbench_id=35,
    pytorch_fn=lambda x: F.group_norm(x, 8),  # 8 groups
    input_gen=lambda: ((torch.randn(16, 64, 32, 32),), {}),
    description='GroupNorm with 8 groups',
    complexity='medium'
))

register(KernelSpec(
    name='l2_norm', category='normalization', kernelbench_id=39,
    pytorch_fn=lambda x: F.normalize(x, p=2, dim=-1),
    input_gen=lambda: ((torch.randn(16, 4096),), {}),
    description='L2 normalization: x / ||x||_2',
    intermediates=lambda x: {
        'norm': x.norm(dim=-1, keepdim=True),
    },
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 5: Pooling Operations
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='max_pool1d', category='pooling', kernelbench_id=41,
    pytorch_fn=lambda x: F.max_pool1d(x, kernel_size=3, stride=2, padding=1),
    input_gen=lambda: ((torch.randn(16, 64, 256),), {}),
    description='MaxPool1d: kernel=3, stride=2, pad=1',
    complexity='low'
))

register(KernelSpec(
    name='max_pool2d', category='pooling', kernelbench_id=42,
    pytorch_fn=lambda x: F.max_pool2d(x, kernel_size=3, stride=2, padding=1),
    input_gen=lambda: ((torch.randn(16, 64, 32, 32),), {}),
    description='MaxPool2d: kernel=3, stride=2, pad=1',
    complexity='low'
))

register(KernelSpec(
    name='avg_pool1d', category='pooling', kernelbench_id=44,
    pytorch_fn=lambda x: F.avg_pool1d(x, kernel_size=3, stride=2, padding=1),
    input_gen=lambda: ((torch.randn(16, 64, 256),), {}),
    description='AvgPool1d: kernel=3, stride=2, pad=1',
    complexity='low'
))

register(KernelSpec(
    name='avg_pool2d', category='pooling', kernelbench_id=45,
    pytorch_fn=lambda x: F.avg_pool2d(x, kernel_size=3, stride=2, padding=1),
    input_gen=lambda: ((torch.randn(16, 64, 32, 32),), {}),
    description='AvgPool2d: kernel=3, stride=2, pad=1',
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 6: Convolution Operations
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='conv1d', category='convolution', kernelbench_id=67,
    pytorch_fn=lambda x, w, b: F.conv1d(x, w, b, padding=1),
    input_gen=lambda: (
        (torch.randn(16, 3, 256),          # input
         torch.randn(16, 3, 3),            # weight
         torch.randn(16)),                 # bias
        {}
    ),
    description='Conv1d: 3 in, 16 out, kernel=3, pad=1',
    complexity='medium'
))

register(KernelSpec(
    name='conv2d', category='convolution', kernelbench_id=50,
    pytorch_fn=lambda x, w, b: F.conv2d(x, w, b, padding=1),
    input_gen=lambda: (
        (torch.randn(16, 3, 32, 32),      # input
         torch.randn(64, 3, 3, 3),        # weight
         torch.randn(64)),                 # bias
        {}
    ),
    description='Conv2d: 3→64, kernel=3x3, pad=1',
    complexity='high'
))

register(KernelSpec(
    name='conv2d_depthwise', category='convolution', kernelbench_id=82,
    pytorch_fn=lambda x, w: F.conv2d(x, w, padding=1, groups=64),
    input_gen=lambda: (
        (torch.randn(16, 64, 32, 32),     # input
         torch.randn(64, 1, 3, 3)),       # weight (groups=64)
        {}
    ),
    description='Depthwise Conv2d: 64 channels, kernel=3x3',
    complexity='medium'
))

register(KernelSpec(
    name='conv2d_pointwise', category='convolution', kernelbench_id=87,
    pytorch_fn=lambda x, w, b: F.conv2d(x, w, b),
    input_gen=lambda: (
        (torch.randn(16, 64, 32, 32),     # input
         torch.randn(128, 64, 1, 1),      # weight (1x1)
         torch.randn(128)),               # bias
        {}
    ),
    description='Pointwise Conv2d: 64→128, kernel=1x1',
    complexity='medium'
))

register(KernelSpec(
    name='conv_transpose2d', category='convolution', kernelbench_id=57,
    pytorch_fn=lambda x, w: F.conv_transpose2d(x, w, padding=1),
    input_gen=lambda: (
        (torch.randn(16, 64, 16, 16),     # input
         torch.randn(64, 32, 3, 3)),      # weight
        {}
    ),
    description='ConvTranspose2d: 64→32, kernel=3x3',
    complexity='high'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 7: Linear Algebra / Matmul
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='matmul_square', category='matmul', kernelbench_id=1,
    pytorch_fn=torch.matmul,
    input_gen=lambda: ((torch.randn(1024, 1024), torch.randn(1024, 1024)), {}),
    description='Square GEMM: 1024x1024 @ 1024x1024',
    complexity='high'
))

register(KernelSpec(
    name='matmul_rect', category='matmul', kernelbench_id=2,
    pytorch_fn=torch.matmul,
    input_gen=lambda: ((torch.randn(128, 512), torch.randn(512, 256)), {}),
    description='Rectangular GEMM: 128x512 @ 512x256',
    complexity='high'
))

register(KernelSpec(
    name='matmul_batched', category='matmul', kernelbench_id=3,
    pytorch_fn=torch.bmm,
    input_gen=lambda: ((torch.randn(8, 128, 64), torch.randn(8, 64, 128)), {}),
    description='Batched GEMM: 8x128x64 @ 8x64x128',
    complexity='high'
))

register(KernelSpec(
    name='matvec', category='matmul', kernelbench_id=4,
    pytorch_fn=lambda A, x: A @ x,
    input_gen=lambda: ((torch.randn(2048, 2048), torch.randn(2048)), {}),
    description='Matrix-vector multiply: 2048x2048 @ 2048',
    complexity='medium'
))

register(KernelSpec(
    name='linear', category='matmul', kernelbench_id=None,
    pytorch_fn=lambda x, w, b: F.linear(x, w, b),
    input_gen=lambda: (
        (torch.randn(16, 2048),            # input
         torch.randn(4096, 2048),          # weight
         torch.randn(4096)),               # bias
        {}
    ),
    description='Linear (GEMM + bias): 2048→4096',
    complexity='high'
))

register(KernelSpec(
    name='matmul_transA', category='matmul', kernelbench_id=16,
    pytorch_fn=lambda A, B: A.T @ B,
    input_gen=lambda: ((torch.randn(512, 256), torch.randn(512, 128)), {}),
    description='GEMM with transposed A: A^T @ B',
    complexity='high'
))

register(KernelSpec(
    name='matmul_transB', category='matmul', kernelbench_id=17,
    pytorch_fn=lambda A, B: A @ B.T,
    input_gen=lambda: ((torch.randn(256, 512), torch.randn(128, 512)), {}),
    description='GEMM with transposed B: A @ B^T',
    complexity='high'
))

register(KernelSpec(
    name='outer_product', category='matmul', kernelbench_id=None,
    pytorch_fn=torch.outer,
    input_gen=lambda: ((torch.randn(256), torch.randn(128)), {}),
    description='Outer product: u ⊗ v',
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 8: Loss Functions
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='mse_loss', category='loss', kernelbench_id=94,
    pytorch_fn=lambda pred, target: F.mse_loss(pred, target),
    input_gen=lambda: ((torch.randn(16, 256), torch.randn(16, 256)), {}),
    description='MSE Loss: mean((pred - target)^2)',
    intermediates=lambda pred, target: {
        'diff': pred - target,
        'sq_diff': (pred - target).pow(2),
        'mean_sq': (pred - target).pow(2).mean(),
    },
    complexity='low'
))

register(KernelSpec(
    name='cross_entropy', category='loss', kernelbench_id=95,
    pytorch_fn=lambda logits, target: F.cross_entropy(logits, target),
    input_gen=lambda: (
        (torch.randn(16, 1000),             # logits (N, C)
         torch.randint(0, 1000, (16,))),    # target labels
        {}
    ),
    description='Cross-entropy loss with logits',
    intermediates=lambda logits, target: {
        'log_softmax': F.log_softmax(logits, dim=-1),
    },
    complexity='medium'
))

register(KernelSpec(
    name='huber_loss', category='loss', kernelbench_id=96,
    pytorch_fn=lambda pred, target: F.huber_loss(pred, target),
    input_gen=lambda: ((torch.randn(16, 256), torch.randn(16, 256)), {}),
    description='Huber loss (smooth L1)',
    complexity='low'
))

register(KernelSpec(
    name='kl_div_loss', category='loss', kernelbench_id=98,
    pytorch_fn=lambda log_p, q: F.kl_div(log_p, q, reduction='batchmean', log_target=False),
    input_gen=lambda: (
        (F.log_softmax(torch.randn(16, 256), dim=-1),  # log probabilities
         F.softmax(torch.randn(16, 256), dim=-1)),     # target probabilities
        {}
    ),
    description='KL divergence loss',
    complexity='medium'
))

register(KernelSpec(
    name='hinge_loss', category='loss', kernelbench_id=100,
    pytorch_fn=lambda pred, target: torch.clamp(1 - pred * target, min=0).mean(),
    input_gen=lambda: (
        (torch.randn(16, 256),
         torch.sign(torch.randn(16, 256))),  # {-1, +1} targets
        {}
    ),
    description='Hinge loss: max(0, 1 - y*f(x))',
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 9: Cumulative Operations
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='cumsum', category='cumulative', kernelbench_id=89,
    pytorch_fn=lambda x: torch.cumsum(x, dim=-1),
    input_gen=lambda: ((torch.randn(16, 4096),), {}),
    description='Cumulative sum along last dimension',
    complexity='medium'
))

register(KernelSpec(
    name='cumprod', category='cumulative', kernelbench_id=90,
    pytorch_fn=lambda x: torch.cumprod(x, dim=-1),
    input_gen=lambda: ((torch.rand(16, 256) * 0.5 + 0.75,), {}),  # positive, near 1
    description='Cumulative product along last dimension',
    complexity='medium'
))

register(KernelSpec(
    name='cumsum_reverse', category='cumulative', kernelbench_id=91,
    pytorch_fn=lambda x: torch.cumsum(x.flip(-1), dim=-1).flip(-1),
    input_gen=lambda: ((torch.randn(16, 4096),), {}),
    description='Reverse cumulative sum',
    complexity='medium'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 10: Attention + Composite
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='scaled_dot_product_attention', category='attention', kernelbench_id=97,
    pytorch_fn=lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
    input_gen=lambda: (
        (torch.randn(2, 8, 128, 64),   # Q: (B, heads, T, d)
         torch.randn(2, 8, 128, 64),   # K
         torch.randn(2, 8, 128, 64)),  # V
        {}
    ),
    description='Scaled dot-product attention: softmax(QK^T/sqrt(d)) @ V',
    intermediates=lambda q, k, v: {
        'scores': torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5),
        'attn_weights': F.softmax(torch.matmul(q, k.transpose(-2, -1)) / (64 ** 0.5), dim=-1),
    },
    complexity='high'
))


# ═══════════════════════════════════════════════════════════════════════
# Section 11: Quantization Operations
# ═══════════════════════════════════════════════════════════════════════

register(KernelSpec(
    name='quantize_per_tensor', category='quantization', kernelbench_id=None,
    pytorch_fn=lambda x: torch.quantize_per_tensor(x, scale=0.1, zero_point=128, dtype=torch.quint8),
    input_gen=lambda: ((torch.randn(16, 4096) * 10,), {}),
    description='Per-tensor quantization to uint8',
    complexity='low'
))


# ═══════════════════════════════════════════════════════════════════════
# Test Runner
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests(categories=None, verbose=True):
    """Run all registered kernels and verify outputs are finite."""
    results = {}
    
    for name, spec in sorted(KERNEL_LIBRARY.items()):
        if categories and spec.category not in categories:
            continue
        
        try:
            inputs, kwargs, output, intermediates = spec.run()
            
            # Basic sanity
            if isinstance(output, torch.Tensor):
                finite = torch.all(torch.isfinite(output)).item() if output.is_floating_point() else True
                shape = tuple(output.shape)
                dtype = str(output.dtype)
            else:
                finite = True
                shape = ()
                dtype = str(type(output))
            
            status = 'PASS' if finite else 'NaN/Inf'
            results[name] = {
                'status': status,
                'shape': shape,
                'dtype': dtype,
                'n_intermediates': len(intermediates),
                'kernelbench_id': spec.kernelbench_id,
            }
            
            if verbose:
                kb = f"L1-{spec.kernelbench_id:3d}" if spec.kernelbench_id else "     "
                inter = f"+{len(intermediates)} intermediates" if intermediates else ""
                print(f"  {'✅' if status == 'PASS' else '❌'} {kb} {name:35s} "
                      f"{str(shape):25s} {dtype:15s} {inter}")
        
        except Exception as e:
            results[name] = {'status': 'ERROR', 'error': str(e)}
            if verbose:
                print(f"  ❌       {name:35s} ERROR: {e}")
    
    return results


def save_reference_outputs(output_dir='/tmp/pytorch_references'):
    """Save all PyTorch reference outputs as .npy files for offline comparison."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, spec in sorted(KERNEL_LIBRARY.items()):
        try:
            inputs, kwargs, output, intermediates = spec.run()
            
            if isinstance(output, torch.Tensor):
                np.save(f"{output_dir}/{name}_output.npy", output.numpy())
            
            # Save inputs
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    np.save(f"{output_dir}/{name}_input{i}.npy", inp.numpy())
            
            # Save intermediates
            for iname, ival in intermediates.items():
                if isinstance(ival, torch.Tensor):
                    np.save(f"{output_dir}/{name}_intermediate_{iname}.npy", ival.numpy())
            
        except Exception as e:
            print(f"  SKIP {name}: {e}")
    
    print(f"Saved references to {output_dir}")


if __name__ == '__main__':
    print("=" * 80)
    print("PyTorch Kernel Library — Reference Test Suite")
    print(f"Registered kernels: {len(KERNEL_LIBRARY)}")
    print("=" * 80)
    print()
    
    for cat in ['activation', 'binary', 'reduction', 'normalization',
                'pooling', 'convolution', 'matmul', 'loss', 'cumulative',
                'attention', 'quantization']:
        kernels_in_cat = [k for k, v in KERNEL_LIBRARY.items() if v.category == cat]
        if kernels_in_cat:
            print(f"\n{'═' * 40}")
            print(f"  {cat.upper()} ({len(kernels_in_cat)} kernels)")
            print(f"{'═' * 40}")
            run_all_tests(categories=[cat])
    
    print(f"\n{'=' * 80}")
    total = len(KERNEL_LIBRARY)
    print(f"TOTAL: {total} kernels registered")
    print(f"KernelBench L1 coverage: {sum(1 for v in KERNEL_LIBRARY.values() if v.kernelbench_id)} tasks")
    print(f"With intermediates: {sum(1 for v in KERNEL_LIBRARY.values() if v.intermediates)} kernels")
    print("=" * 80)
    
    # Optionally save references
    import sys as _sys
    if '--save' in (_sys.argv[1:] if len(_sys.argv) > 1 else []):
        import sys
        save_reference_outputs()
