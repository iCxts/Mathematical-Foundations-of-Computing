import random

def make_diagonally_dominant(n):
    A = [[random.uniform(-10, 10) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        A[i][i] = row_sum + random.uniform(1, 5)
    b = [random.uniform(-10, 10) for _ in range(n)]
    return A, b

def jacobi(A, b, tol=1e-10, max_iter=1000):
    n = len(b)
    x = [0.0] * n
    for k in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new
        if diff < tol:
            print(f"Converged in {k+1} iterations (diff={diff:.2e})")
            return x
    print(f"Did not converge after {max_iter} iterations (diff={diff:.2e})")
    return x

def mat_vec(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def print_system(A, b):
    n = len(b)
    for i in range(n):
        row = "  ".join(f"{A[i][j]:8.3f}" for j in range(n))
        print(f"  [{row}] [x{i}]  {'=' if i == n//2 else ' '}  [{b[i]:8.3f}]")

random.seed(42)

for n in [3, 5, 10]:
    print(f"Test: {n}x{n} system")
    A, b = make_diagonally_dominant(n)
    if n <= 5:
        print_system(A, b)
    x = jacobi(A, b)
    residual = mat_vec(A, x)
    err = max(abs(residual[i] - b[i]) for i in range(n))
    print(f"Solution: {['%.6f' % xi for xi in x]}")
    print(f"Max residual |Ax - b|: {err:.2e}")
