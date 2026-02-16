import numpy as np
import matplotlib.pyplot as plt

loc = np.loadtxt("input/locators.csv", delimiter=",")
d = np.loadtxt("input/approx_dists.csv")

n = len(loc)
xn, yn, dn = loc[-1, 0], loc[-1, 1], d[-1]

A = np.zeros((n - 1, 2))
b = np.zeros(n - 1)

for i in range(n - 1):
    xi, yi, di = loc[i, 0], loc[i, 1], d[i]
    A[i, 0] = 2 * (xn - xi)
    A[i, 1] = 2 * (yn - yi)
    b[i] = di**2 - dn**2 - xi**2 + xn**2 - yi**2 + yn**2

M = A.T @ A
rhs = A.T @ b
pos = np.linalg.solve(M, rhs)

print(f"Battleship position: x = {pos[0]:.4f}, y = {pos[1]:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(loc[:, 0], loc[:, 1], c="blue", s=80, label="Locators", zorder=5)
for i in range(n):
    plt.annotate(f"L{i+1}", (loc[i, 0] + 2, loc[i, 1] + 2), fontsize=8)
plt.scatter(pos[0], pos[1], c="red", s=200, marker="X", label="Battleship", zorder=5)
plt.annotate(f"({pos[0]:.1f}, {pos[1]:.1f})", (pos[0] + 2, pos[1] + 2), fontsize=10, color="red")
plt.legend()
plt.title("Multilateration: Battleship Detection")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.savefig("battleship_plot.png", dpi=150)
plt.show()
