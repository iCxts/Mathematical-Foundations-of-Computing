import numpy as np
import matplotlib.pyplot as plt

# ── define function and gradient (only part that changes per problem) ──────────

def f(p):
    return (p[0]+1)**2 + (p[1]-1)**2 + (p[2]-2)**2

def grad_f(p):
    return np.array([2*(p[0]+1), 2*(p[1]-1), 2*(p[2]-2)])

x0      = np.array([4.0, 5.0, -3.0])
MINIMUM = np.array([-1.0, 1.0, 2.0])

# ── universal GD loop (never changes regardless of dimension) ──────────────────

def gd(f, grad_f, x0, lr=0.1, eps=1e-8, kmax=200):
    x = x0.copy().astype(float)
    dim = len(x)
    history = [x.copy()]

    coord_hdr = "  ".join(f"{'x'+str(i):>10}" for i in range(dim))
    print(f"{'iter':>5}  {coord_hdr}  {'f(x)':>12}  {'|∇f|':>10}  {'|step|':>10}  stop")
    print("-" * (5 + dim*12 + 40))

    for k in range(kmax):
        g      = grad_f(x)
        x_new  = x - lr * g

        step   = np.linalg.norm(x_new - x)
        gnorm  = np.linalg.norm(g)
        f_new  = f(x_new)
        df     = abs(f_new - f(x))

        x = x_new
        history.append(x.copy())

        stop = ""
        if   step  < eps: stop = "step<ε"
        elif gnorm < eps: stop = "|∇f|<ε"
        elif df    < eps: stop = "Δf<ε"
        elif k+1 == kmax: stop = "kmax"

        coords = "  ".join(f"{xi:>10.5f}" for xi in x)
        print(f"{k+1:>5}  {coords}  {f_new:>12.7f}  {gnorm:>10.5f}  {step:>10.7f}  {stop}")

        if stop:
            break

    dist = np.linalg.norm(x - MINIMUM)
    print(f"\n  final x = {np.round(x, 6)}")
    print(f"  f(x)    = {f(x):.2e}  |x − x*| = {dist:.2e}")
    return np.array(history)

# ── run for three learning rates ───────────────────────────────────────────────

learning_rates = [0.1, 0.45, 0.9]
histories = {}

for lr in learning_rates:
    print(f"\n{'═'*70}")
    print(f"  lr = {lr}   x0 = {x0}   true min = {MINIMUM}")
    print(f"{'═'*70}")
    histories[lr] = gd(f, grad_f, x0, lr=lr)

# ── plots ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['steelblue', 'seagreen', 'tomato']
labels = [f'lr={lr}' for lr in learning_rates]
dim = len(x0)
coord_names = [f'x{i}' for i in range(dim)]
true_vals   = MINIMUM

ax = axes[0]
for lr, col, lab in zip(learning_rates, colors, labels):
    h = histories[lr]
    ax.semilogy(np.maximum(h[:, 0]*0 + 1e-16,
                           [f(p) for p in h]), color=col, label=lab)
ax.set_title("f(x) vs iteration (log scale)")
ax.set_xlabel("step"); ax.set_ylabel("f(x)"); ax.legend()

ax = axes[1]
for lr, col, lab in zip(learning_rates, colors, labels):
    h = histories[lr]
    dist = np.linalg.norm(h - MINIMUM, axis=1)
    ax.semilogy(dist, color=col, label=lab)
ax.set_title("|x − x*| vs iteration (log scale)")
ax.set_xlabel("step"); ax.set_ylabel("|x − x*|"); ax.legend()

ax = axes[2]
lr_demo = 0.1
h = histories[lr_demo]
for i, (name, true) in enumerate(zip(coord_names, true_vals)):
    ax.plot(h[:, i], label=f'{name} (→ {true})', color=colors[i])
    ax.axhline(true, linestyle='--', color=colors[i], alpha=0.4)
ax.set_title(f"Coordinate trajectories  (lr={lr_demo})")
ax.set_xlabel("step"); ax.set_ylabel("value"); ax.legend()

plt.tight_layout()
plt.savefig("images/gd_nd.png", dpi=120)
plt.show()
