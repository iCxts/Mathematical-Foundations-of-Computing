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

print("BISECTION METHOD")
a_bis, b_bis = a, b
for i in range(max_iter):
    c = (a_bis + b_bis) / 2
    fc = f(c)
    print(i+1, c, fc)
    if abs(fc) < eps:
        break
    if f(a_bis) * fc < 0:
        b_bis = c
    else:
        a_bis = c

print("NEWTON METHOD")
x = 1.5
for i in range(max_iter):
    fx = f(x)
    print(i+1, x, fx)
    if abs(fx) < eps:
        break
    x = x - fx / df(x)

print("RELAXATION METHOD")
x = 1.5
for i in range(max_iter):
    fx = f(x)
    print(i+1, x, fx)
    if abs(fx) < eps:
        break
    x = alpha * fx + x
