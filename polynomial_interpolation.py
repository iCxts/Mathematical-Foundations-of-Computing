import numpy as np
import matplotlib.pyplot as plt

def interpolate(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    A = np.vander(x, increasing=True)
    c = np.linalg.solve(A, y)
    t = np.linspace(a, b, 500)
    p = sum(c[k] * t**k for k in range(n + 1))
    return x, y, t, p

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

f1 = lambda x: np.sin(x)
x, y, t, p = interpolate(f1, 0, 2 * np.pi, 6)
axes[0, 0].plot(t, p, 'b-')
axes[0, 0].plot(x, y, 'ro', markersize=8)
axes[0, 0].plot(t, f1(t), 'g--')
axes[0, 0].set_title('sin(x), n=6')
axes[0, 0].legend(['polynomial', 'points', 'sin(x)'])

f2 = lambda x: 1 / (1 + 25 * x**2)
x, y, t, p = interpolate(f2, -1, 1, 10)
axes[0, 1].plot(t, p, 'b-')
axes[0, 1].plot(x, y, 'ro', markersize=8)
axes[0, 1].plot(t, f2(t), 'g--')
axes[0, 1].set_title('Runge function 1/(1+25x^2), n=10')
axes[0, 1].legend(['polynomial', 'points', 'f(x)'])

f3 = lambda x: np.exp(-x) * np.cos(3 * x)
x, y, t, p = interpolate(f3, 0, 5, 8)
axes[1, 0].plot(t, p, 'b-')
axes[1, 0].plot(x, y, 'ro', markersize=8)
axes[1, 0].plot(t, f3(t), 'g--')
axes[1, 0].set_title('exp(-x)*cos(3x), n=8')
axes[1, 0].legend(['polynomial', 'points', 'f(x)'])

f4 = lambda x: np.abs(x) + 0.5 * np.sin(4 * x)
x, y, t, p = interpolate(f4, -2, 2, 12)
axes[1, 1].plot(t, p, 'b-')
axes[1, 1].plot(x, y, 'ro', markersize=8)
axes[1, 1].plot(t, f4(t), 'g--')
axes[1, 1].set_title('|x| + 0.5*sin(4x), n=12')
axes[1, 1].legend(['polynomial', 'points', 'f(x)'])

plt.tight_layout()
plt.savefig('interpolation_plots.png', dpi=150)
plt.show()
