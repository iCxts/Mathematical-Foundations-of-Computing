import numpy as np
import matplotlib.pyplot as plt
import csv

def gaussian_elimination(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), b.astype(float).reshape(-1, 1)])
    for col in range(n):
        max_row = col + np.argmax(np.abs(M[col:, col]))
        M[[col, max_row]] = M[[max_row, col]]
        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    return x

def jacobi(A, b, tol=1e-10, max_iter=10000):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A).copy()
    for it in range(1, max_iter + 1):
        x_new = (b - (A @ x - D * x)) / D
        if np.linalg.norm(x_new - x) < tol:
            return x_new, it
        x = x_new
    return x, max_iter

def gauss_seidel(A, b, tol=1e-10, max_iter=10000):
    n = len(b)
    x = np.zeros(n)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            x[i] = (b[i] - np.dot(A[i, :i], x[:i]) - np.dot(A[i, i+1:], x_old[i+1:])) / A[i, i]
        if np.linalg.norm(x - x_old) < tol:
            return x.copy(), it
        x_old = x.copy()
    return x.copy(), max_iter

def random_matrix(n, lo=-5, hi=5, diag_dominant=False):
    A = np.random.uniform(lo, hi, (n, n))
    if diag_dominant:
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i])) - np.abs(A[i, i]) + np.random.uniform(1, 5)
    return A

def hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H

def make_b(A, x_test):
    return A @ x_test

dims = list(range(3, 51))

rows_exp1 = []
for n in dims:
    A = random_matrix(n)
    x_test = np.ones(n)
    b = make_b(A, x_test)
    x = gaussian_elimination(A, b)
    residue = np.linalg.norm(A @ x - b)
    error = np.linalg.norm(x - x_test)
    rows_exp1.append([n, "Gauss", residue, error, ""])

with open("experiment1_random.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "method", "residue", "error", "iterations"])
    w.writerows(rows_exp1)

rows_exp2 = []
for n in dims:
    A = hilbert_matrix(n)
    x_test = np.ones(n)
    b = make_b(A, x_test)
    x_g = gaussian_elimination(A, b)
    rows_exp2.append([n, "Gauss", np.linalg.norm(A @ x_g - b), np.linalg.norm(x_g - x_test), ""])
    x_s, it_s = gauss_seidel(A, b)
    rows_exp2.append([n, "Seidel", np.linalg.norm(A @ x_s - b), np.linalg.norm(x_s - x_test), it_s])

with open("experiment2_hilbert.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "method", "residue", "error", "iterations"])
    w.writerows(rows_exp2)

rows_exp3 = []
for n in dims:
    A = random_matrix(n, diag_dominant=True)
    x_test = np.ones(n)
    b = make_b(A, x_test)
    x_g = gaussian_elimination(A, b)
    rows_exp3.append([n, "Gauss", np.linalg.norm(A @ x_g - b), np.linalg.norm(x_g - x_test), ""])
    x_j, it_j = jacobi(A, b)
    rows_exp3.append([n, "Jacobi", np.linalg.norm(A @ x_j - b), np.linalg.norm(x_j - x_test), it_j])
    x_s, it_s = gauss_seidel(A, b)
    rows_exp3.append([n, "Seidel", np.linalg.norm(A @ x_s - b), np.linalg.norm(x_s - x_test), it_s])

with open("experiment3_diagdom.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "method", "residue", "error", "iterations"])
    w.writerows(rows_exp3)

def plot_experiment(rows, title, filename):
    methods = sorted(set(r[1] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    for m in methods:
        ns = [r[0] for r in rows if r[1] == m]
        residues = [r[2] for r in rows if r[1] == m]
        errors = [r[3] for r in rows if r[1] == m]
        axes[0].plot(ns, residues, 'o-', label=m, markersize=3)
        axes[1].plot(ns, errors, 'o-', label=m, markersize=3)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("||Ax - b||")
    axes[0].set_title("Residue")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("||x - x_test||")
    axes[1].set_title("Distance to exact solution")
    axes[1].legend()
    axes[1].set_yscale("log")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_iterations(rows, title, filename):
    methods = sorted(set(r[1] for r in rows if r[4] != ""))
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        ns = [r[0] for r in rows if r[1] == m]
        iters = [r[4] for r in rows if r[1] == m]
        ax.plot(ns, iters, 'o-', label=m, markersize=3)
    ax.set_xlabel("n")
    ax.set_ylabel("iterations")
    ax.set_title(title + " - Iterations")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

plot_experiment(rows_exp1, "Experiment 1: Random Matrix", "exp1_random.png")
plot_experiment(rows_exp2, "Experiment 2: Hilbert Matrix", "exp2_hilbert.png")
plot_experiment(rows_exp3, "Experiment 3: Diagonally Dominant Matrix", "exp3_diagdom.png")
plot_iterations(rows_exp2, "Experiment 2: Hilbert Matrix", "exp2_hilbert_iters.png")
plot_iterations(rows_exp3, "Experiment 3: Diagonally Dominant Matrix", "exp3_diagdom_iters.png")

print("done")
