import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**4 - 4*x**2 + x

def df(x):
    return 4*x**3 - 8*x + 1

def gradient_descent(x0, lr, steps):
    x = x0
    history = [x]
    print(f"{'step':>4}  {'x':>10}  {'f(x)':>12}  {'df(x)':>12}")
    print("-" * 46)
    for i in range(steps):
        grad = df(x)
        x = x - lr * grad
        history.append(x)
        print(f"{i+1:>4}  {x:>10.6f}  {f(x):>12.6f}  {grad:>12.6f}")
    return np.array(history)

x0, lr, steps = 2.0, 0.05, 30
print(f"start x={x0}, lr={lr}\n")
history = gradient_descent(x0, lr, steps)

xs = np.linspace(-2.5, 2.5, 400)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(xs, f(xs), 'b', linewidth=2)
axes[0].scatter(history, f(history), c=range(len(history)), cmap='plasma', zorder=5, s=50)
axes[0].plot(history, f(history), 'k--', alpha=0.3)
axes[0].set_title("Gradient descent path on f(x) = x⁴ − 4x² + x")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

axes[1].plot(range(len(history)), f(history), 'o-', color='steelblue', markersize=4)
axes[1].set_title("f(x) vs iteration")
axes[1].set_xlabel("step")
axes[1].set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("images/gradient_descent_1d.png", dpi=120)
plt.show()

print(f"\nfinal x = {history[-1]:.6f},  f(x) = {f(history[-1]):.6f}")
