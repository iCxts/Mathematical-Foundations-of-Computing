import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from scipy.sparse import eye, kron, diags, tril
from scipy.sparse.linalg import spsolve, spsolve_triangular

# ── 1. Load & resize ──────────────────────────────────────────────────────────
image = io.imread("monkey.jpeg")
image = transform.resize(image, (128, 128), anti_aliasing=True)
gray  = color.rgb2gray(image)

# ── 2. Add noise ──────────────────────────────────────────────────────────────
np.random.seed(42)
noisy = gray + 0.15 * np.random.randn(*gray.shape)
noisy = np.clip(noisy, 0, 1)

# ── 3. Build sparse system  M = I + λL ────────────────────────────────────────
H, W = gray.shape
Dh = kron(eye(H), diags([-np.ones(W-1), np.ones(W-1)], [0, 1], shape=(W-1, W)))
Dv = kron(diags([-np.ones(H-1), np.ones(H-1)], [0, 1], shape=(H-1, H)), eye(W))
L  = Dh.T @ Dh + Dv.T @ Dv

lam = 3.0
M   = eye(H * W) + lam * L
b   = noisy.flatten()


# ── 4. Solvers ────────────────────────────────────────────────────────────────

def jacobi(M, b, max_iter=500, tol=1e-5):
    d = M.diagonal()
    R = M - diags(d)
    x = b.copy()
    residuals = []
    for _ in range(max_iter):
        x   = (b - R @ x) / d
        res = np.linalg.norm(M @ x - b)
        residuals.append(res)
        if res < tol:
            break
    return x, residuals


def gauss_seidel(M, b, max_iter=500, tol=1e-5):
    M_csr = M.tocsr()
    L_tri = tril(M_csr).tocsr()
    U_tri = (M_csr - L_tri).tocsr()
    x = b.copy()
    residuals = []
    for _ in range(max_iter):
        x   = spsolve_triangular(L_tri, b - U_tri @ x, lower=True)
        res = np.linalg.norm(M_csr @ x - b)
        residuals.append(res)
        if res < tol:
            break
    return x, residuals


# ── 5. Run all three ──────────────────────────────────────────────────────────
print("Running Jacobi ...")
x_jacobi, res_jacobi = jacobi(M, b)

print("Running Gauss-Seidel ...")
x_gs, res_gs = gauss_seidel(M, b)

print("Running Direct (Gaussian Elimination) ...")
x_direct = spsolve(M, b)

print(f"\nJacobi         : {len(res_jacobi)} iters   residual = {res_jacobi[-1]:.2e}")
print(f"Gauss-Seidel   : {len(res_gs)} iters   residual = {res_gs[-1]:.2e}")
print(f"Direct (GE)    : residual = {np.linalg.norm(M @ x_direct - b):.2e}")


# ── 6. Plot: Original | Noisy | Jacobi | Gauss-Seidel | Direct ───────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
imgs   = [
    gray,
    noisy,
    x_jacobi.reshape(H, W),
    x_gs.reshape(H, W),
    x_direct.reshape(H, W),
]
titles = [
    "Original",
    "Corrupted\n(noisy)",
    f"Jacobi\n({len(res_jacobi)} iterations)",
    f"Gauss-Seidel\n({len(res_gs)} iterations)",
    "Direct\n(Gaussian Elimination)",
]
for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
plt.suptitle("Image Denoising — Solving  (I + λL) x = b", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("../images/denoising_comparison.png", dpi=150, bbox_inches="tight")
plt.show()


# ── 7. Convergence plot ───────────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
plt.semilogy(res_jacobi, label=f"Jacobi  ({len(res_jacobi)} iters)")
plt.semilogy(res_gs,     label=f"Gauss-Seidel  ({len(res_gs)} iters)")
plt.xlabel("Iteration")
plt.ylabel("Residual  ‖Mx − b‖")
plt.title("Convergence of Iterative Solvers")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("../images/denoising_convergence.png", dpi=150)
plt.show()


# ── 8. Condition number demo — effect of λ ────────────────────────────────────
lam_values  = [0.1, 1.0, 3.0, 10.0]
fig, axes   = plt.subplots(1, len(lam_values) + 1, figsize=(20, 4))
axes[0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("Noisy input", fontsize=11)
axes[0].axis("off")
for ax, lv in zip(axes[1:], lam_values):
    M_lv = eye(H * W) + lv * L
    x_lv = spsolve(M_lv, b)
    ax.imshow(np.clip(x_lv.reshape(H, W), 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"λ = {lv}", fontsize=11)
    ax.axis("off")
plt.suptitle("Effect of λ  (small = noisy,  large = over-smoothed)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("../images/denoising_lambda.png", dpi=150, bbox_inches="tight")
plt.show()
