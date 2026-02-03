import math

def g(x):
    return math.sqrt(x + 2)

x = 3.0
tol = 0.001
max_iter = 20

for k in range(max_iter):
    x_new = g(x)
    print(f"k={k+1}: x = {x_new}")
    if abs(x_new - x) < tol:
        break
    x = x_new

print(f"\nFixed point: {x_new}")
