import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 40
x_raw = np.linspace(0, 10, n)
TRUE_C0, TRUE_C1 = 2.0, 1.5
y = TRUE_C0 + TRUE_C1 * x_raw + np.random.randn(n) * 1.5

x_mean, x_std = x_raw.mean(), x_raw.std()
x_norm = (x_raw - x_mean) / x_std

H_raw  = np.array([[2, 2*np.mean(x_raw)],  [2*np.mean(x_raw),  2*np.mean(x_raw**2)]])
H_norm = np.array([[2, 2*np.mean(x_norm)], [2*np.mean(x_norm), 2*np.mean(x_norm**2)]])
eigs_raw  = np.linalg.eigvalsh(H_raw)
eigs_norm = np.linalg.eigvalsh(H_norm)

print("=== Condition numbers ===")
print(f"  Raw x:        κ = {eigs_raw.max()/eigs_raw.min():.1f}   (max stable lr < {2/eigs_raw.max():.4f})")
print(f"  Normalized x: κ = {eigs_norm.max()/eigs_norm.min():.1f}   (max stable lr < {2/eigs_norm.max():.4f})")

def make_fns(x):
    def loss(c0, c1): return np.mean((y - c0 - c1 * x) ** 2)
    def grad(c0, c1):
        r = y - c0 - c1 * x
        return -2 * np.mean(r), -2 * np.mean(x * r)
    return loss, grad

def gd(loss, grad, lr, eps=1e-6, kmax=200):
    c0, c1 = 0.0, 0.0
    history = [(c0, c1, loss(c0, c1))]
    print(f"\n{'':=<65}")
    print(f"  lr={lr}  kmax={kmax}  eps={eps}")
    print(f"{'':=<65}")
    print(f"{'iter':>5}  {'c0':>9}  {'c1':>9}  {'loss':>11}  {'|grad|':>9}  {'|step|':>9}  stop")
    print(f"{'':->65}")

    for k in range(kmax):
        dc0, dc1 = grad(c0, c1)
        c0_new = c0 - lr * dc0
        c1_new = c1 - lr * dc1

        step  = np.hypot(c0_new - c0, c1_new - c1)
        gnorm = np.hypot(dc0, dc1)
        l_new = loss(c0_new, c1_new)
        dl    = abs(l_new - history[-1][2])

        c0, c1 = c0_new, c1_new
        history.append((c0, c1, l_new))

        stop = ""
        if   step  < eps: stop = "step<ε"
        elif gnorm < eps: stop = "|∇f|<ε"
        elif dl    < eps: stop = "Δloss<ε"
        elif k + 1 == kmax: stop = "kmax"

        print(f"{k+1:>5}  {c0:>9.5f}  {c1:>9.5f}  {l_new:>11.6f}  {gnorm:>9.5f}  {step:>9.6f}  {stop}")

        if stop:
            break

    print(f"\n  → c0={c0:.5f}  c1={c1:.5f}")
    return np.array(history)

loss_raw,  grad_raw  = make_fns(x_raw)
loss_norm, grad_norm = make_fns(x_norm)

print("\n\n>>> RAW x: slow convergence due to high condition number")
h_slow   = gd(loss_raw,  grad_raw,  lr=0.001)
h_opt    = gd(loss_raw,  grad_raw,  lr=0.025)
h_div    = gd(loss_raw,  grad_raw,  lr=0.035, kmax=10)

print("\n\n>>> NORMALIZED x: fast convergence (κ ≈ 1)")
h_fast   = gd(loss_norm, grad_norm, lr=0.24)

# recover original-space params from normalized run
c0_norm, c1_norm = h_fast[-1, 0], h_fast[-1, 1]
c1_orig = c1_norm / x_std
c0_orig = c0_norm - c1_norm * x_mean / x_std
print(f"\n  Recovered original params: c0={c0_orig:.5f}  c1={c1_orig:.5f}  (true: c0={TRUE_C0}, c1={TRUE_C1})")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
ax.scatter(x_raw, y, s=25, color='gray', zorder=5, label='data')
ax.plot(x_raw, TRUE_C0 + TRUE_C1 * x_raw, 'k--', lw=2, label=f'true')
ax.plot(x_raw, h_slow[-1,0] + h_slow[-1,1]*x_raw, color='steelblue', lw=1.5, label=f'lr=0.001 ({len(h_slow)-1} steps)')
ax.plot(x_raw, h_opt[-1,0]  + h_opt[-1,1]*x_raw,  color='orange',    lw=1.5, label=f'lr=0.025 ({len(h_opt)-1} steps)')
ax.plot(x_raw, c0_orig + c1_orig*x_raw,            color='seagreen',  lw=1.5, label=f'normalized ({len(h_fast)-1} steps)')
ax.set_title("Fitted lines"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.semilogy(h_slow[:, 2], color='steelblue', label='lr=0.001 (raw x)')
ax.semilogy(h_opt[:, 2],  color='orange',    label='lr=0.025 (raw x)')
ax.semilogy(h_fast[:, 2], color='seagreen',  label='lr=0.24 (norm x)')
ax.set_title("Loss vs iteration (log scale)"); ax.set_xlabel("step"); ax.set_ylabel("MSE"); ax.legend()

ax = axes[1, 0]
c0g = np.linspace(-0.5, 3.5, 200)
c1g = np.linspace(0.5, 2.5, 200)
C0, C1 = np.meshgrid(c0g, c1g)
L = np.vectorize(loss_raw)(C0, C1)
ax.contourf(C0, C1, L, levels=40, cmap='Blues_r')
ax.contour(C0, C1, L, levels=40, colors='white', linewidths=0.3, alpha=0.4)
ax.plot(h_slow[:, 0], h_slow[:, 1], 'o-', color='steelblue', ms=2, lw=1, label='lr=0.001')
ax.plot(h_opt[:, 0],  h_opt[:, 1],  'o-', color='orange',    ms=2, lw=1, label='lr=0.025')
ax.plot(TRUE_C0, TRUE_C1, 'w*', ms=14, zorder=10, label='true min')
ax.set_title("Loss landscape (raw x) — elongated bowl due to κ≈140")
ax.set_xlabel("c0"); ax.set_ylabel("c1"); ax.legend(fontsize=8)

ax = axes[1, 1]
c0g2 = np.linspace(-0.5, 3.0, 200)
c1g2 = np.linspace(-1, 4, 200)
C0n, C1n = np.meshgrid(c0g2, c1g2)
Ln = np.vectorize(loss_norm)(C0n, C1n)
ax.contourf(C0n, C1n, Ln, levels=40, cmap='Greens_r')
ax.contour(C0n, C1n, Ln, levels=40, colors='white', linewidths=0.3, alpha=0.4)
ax.plot(h_fast[:, 0], h_fast[:, 1], 'o-', color='seagreen', ms=3, lw=1.5, label='lr=0.24 (norm x)')
c0_true_norm = TRUE_C0 + TRUE_C1 * x_mean / x_std * 0
ax.set_title("Loss landscape (normalized x) — round bowl, fast GD")
ax.set_xlabel("c0 (norm)"); ax.set_ylabel("c1 (norm)"); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("images/linear_regression_gd.png", dpi=120)
plt.show()
