import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import convolve
from scipy.linalg import toeplitz, solve_triangular

image = io.imread("images/astronaut.jpg")
gray = color.rgb2gray(image)

np.random.seed(42)
sigma, k = 1.5, 3
t = np.arange(-k, k + 1, dtype=float)
h = np.exp(-t ** 2 / (2 * sigma ** 2))
h /= h.sum()
kernel_2d = np.outer(h, h)

blurred = convolve(gray, kernel_2d)
corrupted = blurred + 0.03 * np.random.randn(*gray.shape)
corrupted = np.clip(corrupted, 0, 1)

n = 32
row0, col0 = 220, 220
patch_orig    = gray[row0:row0+n, col0:col0+n]
patch_corrupt = corrupted[row0:row0+n, col0:col0+n]

first_col = np.zeros(n)
first_col[:k+1] = h[k::-1]
first_row = np.zeros(n)
first_row[:k+1] = h[k:]
A1d = toeplitz(first_col, first_row)
A   = np.kron(A1d, A1d)

b   = patch_corrupt.flatten()
lam = 0.05
M   = A.T @ A + lam * np.eye(n * n)
rhs = A.T @ b


def jacobi(M, rhs, max_iter=500, tol=1e-8):
    x = np.zeros(len(rhs))
    d = np.diag(M)
    R = M - np.diag(d)
    residuals = []
    for _ in range(max_iter):
        x = (rhs - R @ x) / d
        res = np.linalg.norm(M @ x - rhs)
        residuals.append(res)
        if res < tol:
            break
    return x, residuals


def gauss_seidel(M, rhs, max_iter=500, tol=1e-8):
    L = np.tril(M)
    U = M - L
    x = np.zeros(len(rhs))
    residuals = []
    for _ in range(max_iter):
        x = solve_triangular(L, rhs - U @ x, lower=True)
        res = np.linalg.norm(M @ x - rhs)
        residuals.append(res)
        if res < tol:
            break
    return x, residuals


x_jacobi, res_jacobi = jacobi(M, rhs)
x_gs,     res_gs     = gauss_seidel(M, rhs)
x_direct             = np.linalg.solve(M, rhs)

print(f"Jacobi       : {len(res_jacobi)} iters,  residual = {res_jacobi[-1]:.2e}")
print(f"Gauss-Seidel : {len(res_gs)} iters,  residual = {res_gs[-1]:.2e}")
print(f"Direct solve : residual = {np.linalg.norm(M @ x_direct - rhs):.2e}")

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
imgs   = [patch_orig, patch_corrupt,
          x_jacobi.reshape(n, n), x_gs.reshape(n, n), x_direct.reshape(n, n)]
titles = ["Original", "Corrupted", "Jacobi", "Gauss-Seidel", "Direct  (Gauss Elim)"]
for ax, img, title in zip(axes, imgs, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("images/restoration_comparison.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))
plt.semilogy(res_jacobi, label="Jacobi")
plt.semilogy(res_gs,     label="Gauss-Seidel")
plt.xlabel("Iteration")
plt.ylabel("Residual  ||Mx - b||")
plt.title("Convergence of Iterative Solvers")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("images/convergence.png", dpi=150)
plt.show()

cond_unreg = np.linalg.cond(A.T @ A)
cond_reg   = np.linalg.cond(M)
print(f"Condition number  without regularization : {cond_unreg:.2e}")
print(f"Condition number  with    lam = {lam}      : {cond_reg:.2e}")

M_unreg = A.T @ A + 1e-10 * np.eye(n * n)
x_unreg = np.linalg.solve(M_unreg, rhs)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(x_direct.reshape(n, n), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"With regularization  (lam={lam})")
axes[0].axis("off")
axes[1].imshow(np.clip(x_unreg.reshape(n, n), 0, 1), cmap="gray")
axes[1].set_title("Without regularization  (lam ~ 0)")
axes[1].axis("off")
plt.suptitle("Why condition number matters")
plt.tight_layout()
plt.savefig("images/condition_number.png", dpi=150)
plt.show()
