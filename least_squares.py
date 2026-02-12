import numpy as np
import matplotlib.pyplot as plt

n = 100
true_c0, true_c1 = 3.0, 2.0

x = np.random.uniform(-10, 10, n)
y = true_c0 + true_c1 * x + np.random.normal(0, 2, n)

A = np.column_stack([np.ones(n), x])
M = A.T @ A
rhs = A.T @ y
c = np.linalg.solve(M, rhs)

print(f"true: c0={true_c0}, c1={true_c1}")
print(f"found: c0={c[0]:.4f}, c1={c[1]:.4f}")

xs = np.linspace(x.min(), x.max(), 200)
plt.scatter(x, y, s=10)
plt.plot(xs, c[0] + c[1] * xs, 'r')
plt.title(f"y = {c[0]:.2f} + {c[1]:.2f}x")
plt.show()
