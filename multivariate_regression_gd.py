import numpy as np


# Step 1: original code, works as is (3D, no noise)

def test_plane(x):
    return 2 * x[0] - x[1] + 3*x[2]


n = 1000
m = 3

a = -1
b = 1

x_vals = np.random.uniform(a, b, size=(m, n))

y_vals_exact = test_plane(x_vals)

noise = np.random.normal(0, 0.5, size=y_vals_exact.shape)
noise = np.zeros_like(y_vals_exact)
y = y_vals_exact + noise

X = np.stack(x_vals, axis=1)

c = np.array([1.0, 1.0, 1.0])

max_iters = 10000
alpha = 0.001
eps = 0.000001

for i in range(max_iters):
    c -= alpha * X.T @ (X @ c - y)
    loss = np.linalg.norm(X @ c - y) ** 2
    step = alpha * X.T @ (X @ c - y)
    if np.linalg.norm(step) < eps:
        break
    print(f"{i}: loss: {loss:.2f}; c = {c}")

print(f"Final: loss: {loss:.6f}; c = {c}")
print(f"True c = [2, -1, 3]")


# Step 2 + 3: noise added, dimensionality increased to 5

print("\n--- 5D with noise ---\n")

def test_plane_5d(x):
    return 2*x[0] - x[1] + 3*x[2] - 0.5*x[3] + 1.5*x[4]


m = 5

x_vals = np.random.uniform(a, b, size=(m, n))
y_vals_exact = test_plane_5d(x_vals)

noise = np.random.normal(0, 0.5, size=y_vals_exact.shape)
y = y_vals_exact + noise

X = np.stack(x_vals, axis=1)

c = np.ones(m)

for i in range(max_iters):
    c -= alpha * X.T @ (X @ c - y)
    loss = np.linalg.norm(X @ c - y) ** 2
    step = alpha * X.T @ (X @ c - y)
    if np.linalg.norm(step) < eps:
        break
    print(f"{i}: loss: {loss:.2f}; c = {c}")

print(f"Final ({i} iters): loss: {loss:.4f}")
print(f"recovered c = {np.round(c, 4)}")
print(f"true     c = [2, -1, 3, -0.5, 1.5]")
