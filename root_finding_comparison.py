import math

def f(x):
    return x**3 - x - 2

def df(x):
    return 3*x**2 - 1

a = 1.0
b = 2.0
alpha = -0.18
eps = 1e-6
max_iter = 100

print("="*60)
print("BISECTION METHOD")
print("="*60)
a_bis, b_bis = a, b
for i in range(max_iter):
    c = (a_bis + b_bis) / 2
    fc = f(c)
    print(f"iter {i+1:3d}: x = {c:.10f}, f(x) = {fc:.2e}")
    if abs(fc) < eps:
        break
    if f(a_bis) * fc < 0:
        b_bis = c
    else:
        a_bis = c

print("\n" + "="*60)
print("NEWTON METHOD")
print("="*60)
x = 1.5
for i in range(max_iter):
    fx = f(x)
    print(f"iter {i+1:3d}: x = {x:.10f}, f(x) = {fx:.2e}")
    if abs(fx) < eps:
        break
    x = x - fx / df(x)

print("\n" + "="*60)
print("RELAXATION METHOD (alpha = " + str(alpha) + ")")
print("="*60)
x = 1.5
for i in range(max_iter):
    fx = f(x)
    print(f"iter {i+1:3d}: x = {x:.10f}, f(x) = {fx:.2e}")
    if abs(fx) < eps:
        break
    x = alpha * fx + x
