"""Shakedown: verify one-way-leaky semantics before burning GPU time.
Run-not-assume applied to our own backward pass."""
import sys, torch
sys.path.insert(0, '.')
from activations import strict_relu, one_way_leaky

torch.manual_seed(0)
x = torch.randn(4096, dtype=torch.float64, requires_grad=True)
# Plant exact zeros and exact-boundary values (the lattice points of interest):
with torch.no_grad():
    x[::7] = 0.0

# ── 1. strict vs torch native, forward + backward ──
g = torch.randn_like(x)
y1 = strict_relu(x); (y1 * g).sum().backward()
g1 = x.grad.clone(); x.grad = None
y2 = torch.relu(x); (y2 * g).sum().backward()
g2 = x.grad.clone(); x.grad = None
assert torch.equal(y1, y2), "forward mismatch"
assert torch.equal(g1, g2), "strict backward != torch native (incl x=0)"
print("✓ 1. strict_relu == torch.relu forward+backward, bit-exact (zeros planted)")

# ── 2. oneway(alpha=0) == strict ──
y3 = one_way_leaky(x, 0.0); (y3 * g).sum().backward()
g3 = x.grad.clone(); x.grad = None
assert torch.equal(g3, g1), "alpha=0 degenerate must equal strict"
print("✓ 2. oneway(0) == strict relu (degenerate anchor)")

# ── 3. decomposition: oneway == strict + alpha * revival-channel ──
a = 0.01
y4 = one_way_leaky(x, a); (y4 * g).sum().backward()
g4 = x.grad.clone(); x.grad = None
revival = ((x <= 0) & (g < 0)).to(torch.float64)
expected = g * ((x > 0).to(torch.float64) + a * revival)
assert torch.allclose(g4, expected, atol=0, rtol=0), "revival decomposition"
print(f"✓ 3. oneway decomposes exactly: dead+revival elements = {int(revival.sum())} of {len(x)}")

# ── 4. semantic probe: a dead unit the loss wants LARGER gets gradient; one it wants smaller gets none ──
xx = torch.tensor([-1.0, -1.0], dtype=torch.float64, requires_grad=True)
yy = one_way_leaky(xx, a)
# loss = -y[0] + y[1]: wants y0 LARGER (dL/dy0=-1<0 revival), y1 smaller (dL/dy1=+1, no flow)
loss = -yy[0] + yy[1]
loss.backward()
assert xx.grad[0] == -a, f"revival gradient wrong: {xx.grad[0]}"
assert xx.grad[1] == 0.0, f"suppression must be blocked: {xx.grad[1]}"
print(f"✓ 4. semantics: under-contributing dead unit gets {float(xx.grad[0]):+.3f} (can revive); over-contributing gets 0 (already at floor)")

# ── 5. x=0 EXACTLY, both signs ──
xz = torch.zeros(2, dtype=torch.float64, requires_grad=True)
yz = one_way_leaky(xz, a)
(-yz[0] + yz[1]).backward()
assert xz.grad[0] == -a and xz.grad[1] == 0.0
print("✓ 5. at exactly x=0: revival side leaks alpha, suppression side 0 (argmin over Clarke [0,1] scaled)")

print("\nALL SHAKEDOWN CHECKS PASS — sign orientation verified against torch convention")
