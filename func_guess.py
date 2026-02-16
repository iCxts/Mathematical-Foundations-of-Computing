import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("input/func_guess_data.csv", delimiter=",")
x = data[:, 0]
y = data[:, 1]

A = np.column_stack([x, x**2, np.sin(x), np.cos(x), np.exp(x) / 100])
M = A.T @ A
rhs = A.T @ y
c = np.linalg.solve(M, rhs)

c_round = np.round(c).astype(int)
print("Exact coefficients:", c)
print("Rounded coefficients:", c_round)
print(f"g(x) = {c_round[0]}*x + {c_round[1]}*x^2 + {c_round[2]}*sin(x) + {c_round[3]}*cos(x) + {c_round[4]}*e^x/100")

xs = np.linspace(1, 10, 500)
ys = c_round[0]*xs + c_round[1]*xs**2 + c_round[2]*np.sin(xs) + c_round[3]*np.cos(xs) + c_round[4]*np.exp(xs)/100

plt.scatter(x, y, s=10, label="data")
plt.plot(xs, ys, "r", linewidth=2, label="fit (rounded)")
plt.legend()
plt.title(f"g(x) = {c_round[0]}x + {c_round[1]}x² + {c_round[2]}sin(x) + {c_round[3]}cos(x) + {c_round[4]}eˣ/100")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("func_guess_plot.png", dpi=150)
plt.show()
