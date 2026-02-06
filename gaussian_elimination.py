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


A = [[1, 1, 1], [3, 2, 1], [2, -1, 4]]
b = [6, 10, 12]
x = gaussian_elimination(A, b)
print(f"Test 1: x = {x}")

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
b = [8, -11, -3]
x = gaussian_elimination(A, b)
print(f"Test 2: x = {x}")

A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
b = [6, 15, 25]
x = gaussian_elimination(A, b)
print(f"Test 3: x = {x}")
