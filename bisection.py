def f(x):
    return x**2 -4

a = 0
b = 2

for i in range(100):
    c = (a + b) / 2
    print(c)
    if f(a) * f(c) < 0:
        b = c
    else:
        a = c
