import copy

def gaussian_elimination(A, b):
    n = len(A)

    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x


def hilbert_matrix(n):
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def compute_b(A, x):
    n = len(A)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def max_error(x_computed, x_expected):
    return max(abs(x_computed[i] - x_expected[i]) for i in range(len(x_expected)))


dimensions = [5, 10, 20, 50, 100]

for n in dimensions:
    H = hilbert_matrix(n)
    x_expected = [1.0] * n
    b = compute_b(H, x_expected)

    H_copy = copy.deepcopy(H)
    b_copy = copy.deepcopy(b)

    x_computed = gaussian_elimination(H_copy, b_copy)

    error = max_error(x_computed, x_expected)

    print(f"n = {n:3d} | Max error: {error:.6e}")
    if n <= 10:
        print(f"         Solution: {[round(xi, 6) for xi in x_computed]}")
    print()
