"""One-way-leaky ReLU — sign-dependent backward convention (Heath, 2026-11-15).

Five-arm factorial of backward conventions over an IDENTICAL forward ReLU
(except the two true-leaky arms, whose forward leaks by definition):

  arm 1: relu          forward max(0,x); backward (x>0)           [control]
  arm 2: leaky(a)      forward x>0?x:ax; backward (x>0)+a(x<=0)   [torch native]
  arm 3: leaky(a/2)    same at half slope
  arm 4: oneway(a)     forward max(0,x); backward (x>0) + a*(x<=0)*REVIVAL
  arm 5: oneway(a/2)   same at half slope

REVIVAL is the sign test: gradient passes through a dead unit ONLY when the
loss asks the unit's output to be LARGER.

Sign orientation (the classic silent-wrong spot, stated explicitly):
torch convention: backward receives grad_out = dL/dy. Gradient DESCENT steps
w -= lr * dL/dw. The loss wants y LARGER exactly when dL/dy < 0 (increasing
y decreases L). Therefore REVIVAL mask = (grad_out < 0) on x<=0 territory.
Section 7.3 of bpd/docs/relu_gradient_at_zero.md: at exactly x=0 this is the
argmin over the Clarke subdifferential [0,1]; for x<0 it is a finite-step
(straight-through-style) estimator oriented opposite the forward rectifier:
dead units can be pulled toward life, never pushed deeper into death.
"""
import torch


class OneWayLeakyReLU(torch.autograd.Function):
    """Forward EXACTLY ReLU; backward leaks alpha on dead units only when
    the upstream gradient favors revival (grad_out < 0)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.clamp(x, min=0.0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alive = (x > 0).to(grad_out.dtype)
        revival = ((x <= 0) & (grad_out < 0)).to(grad_out.dtype)
        return grad_out * (alive + ctx.alpha * revival), None


def one_way_leaky(x, alpha=0.01):
    return OneWayLeakyReLU.apply(x, alpha)


class LeakyFwdGatedBwd(torch.autograd.Function):
    """Heath's fourth corner (2026-11-18): forward LEAKY (negative side
    transmits alpha*x), backward sign-gated like one-way (full g when
    x>0; alpha*g on the inactive side ONLY in the revival direction;
    suppression blocked). Completes the {fwd: strict,leaky} x
    {bwd: standard,gated} square: relu=(strict,strict-bwd),
    leaky=(leaky,leaky-bwd), oneway=(strict,gated), lwgate=(leaky,gated)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return torch.nn.functional.leaky_relu(x, alpha)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alive = (x > 0).to(grad_out.dtype)
        revival = ((x <= 0) & (grad_out < 0)).to(grad_out.dtype)
        return grad_out * (alive + ctx.alpha * revival), None


def leaky_fwd_gated(x, alpha=0.01):
    return LeakyFwdGatedBwd.apply(x, alpha)


class StrictReLU(torch.autograd.Function):
    """Control arm pinned to the strict (x>0) backward mask — the documented
    kernel contract (relu doc section 6) — independent of any torch-internal
    convention drift."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(x, min=0.0)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out * (x > 0).to(grad_out.dtype)


def strict_relu(x):
    return StrictReLU.apply(x)


def make_activation(arm: str):
    """arm in: relu, leaky_10, leaky_5, oneway_10, oneway_5
    (suffix = alpha in thousandths: 10 -> 0.010, 5 -> 0.005)"""
    if arm == 'relu':
        return strict_relu
    kind, millis = arm.rsplit('_', 1)
    alpha = float(millis) / 1000.0  # supports fractional thousandths, e.g. oneway_2.5 -> 0.0025
    if kind == 'leaky':
        return lambda x: torch.nn.functional.leaky_relu(x, alpha)
    if kind == 'oneway':
        return lambda x: one_way_leaky(x, alpha)
    if kind == 'lwgate':
        return lambda x: leaky_fwd_gated(x, alpha)
    raise ValueError(arm)


ARMS = ['relu', 'leaky_10', 'leaky_5', 'oneway_10', 'oneway_5']
