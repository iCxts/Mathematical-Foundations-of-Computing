import math
import matplotlib.pyplot as plt

tests = [
    {
        "name": "x^2 - 4",
        "f": lambda x: x**2 - 4,
        "df": lambda x: 2 * x,
        "a": 0.0,
        "b": 5.0,
        "x0": 5.0,
        "alpha": -0.1,
    },
    {
        "name": "cos(x) - x",
        "f": lambda x: math.cos(x) - x,
        "df": lambda x: -math.sin(x) - 1,
        "a": 0.0,
        "b": 2.0,
        "x0": 2.0,
        "alpha": 0.4,
    },
    {
        "name": "(x - 1)^3",
        "f": lambda x: (x - 1) ** 3,
        "df": lambda x: 3 * (x - 1) ** 2,
        "a": 0.0,
        "b": 3.0,
        "x0": 3.0,
        "alpha": None,
    },
]

tol = 1e-10
max_iter = 100

fig, axes = plt.subplots(1, len(tests), figsize=(6 * len(tests), 5))
if len(tests) == 1:
    axes = [axes]

for idx, test in enumerate(tests):
    ax = axes[idx]
    f = test["f"]
    df = test["df"]

    bisection_errors = []
    a_bis, b_bis = test["a"], test["b"]
    for i in range(max_iter):
        c = (a_bis + b_bis) / 2
        bisection_errors.append(b_bis - a_bis)
        if b_bis - a_bis < tol:
            break
        if f(a_bis) * f(c) < 0:
            b_bis = c
        else:
            a_bis = c

    newton_errors = []
    x = test["x0"]
    for i in range(max_iter):
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x_new = x - f(x) / dfx
        newton_errors.append(abs(x_new - x))
        if abs(x_new - x) < tol:
            break
        x = x_new

    fp_errors = []
    if test["alpha"] is not None:
        alpha = test["alpha"]
        x = test["x0"]
        for i in range(max_iter):
            x_new = alpha * f(x) + x
            fp_errors.append(abs(x_new - x))
            if abs(x_new - x) < tol:
                break
            x = x_new

    ax.plot(range(len(bisection_errors)), bisection_errors, "b-o", markersize=3, label="Bisection")
    ax.plot(range(len(newton_errors)), newton_errors, "r-s", markersize=3, label="Newton")
    if fp_errors:
        ax.plot(range(len(fp_errors)), fp_errors, "g-^", markersize=3, label="Fixed Point")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_title(test["name"])
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
