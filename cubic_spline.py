import numpy as np
import matplotlib.pyplot as plt

x_nodes = [0, 1, 2]
y_nodes = [1, 2, 0]

# S0(x) = a0 + b0*x + c0*x^2 + d0*x^3             on [0,1]
# S1(x) = a1 + b1*(x-1) + c1*(x-1)^2 + d1*(x-1)^3 on [1,2]
#
# 8 unknowns: a0,b0,c0,d0, a1,b1,c1,d1
# 8 equations:
#   1) S0(0)=1        => a0 = 1
#   2) S0(1)=2        => a0+b0+c0+d0 = 2
#   3) S1(1)=2        => a1 = 2
#   4) S1(2)=0        => a1+b1+c1+d1 = 0
#   5) S0'(1)=S1'(1)  => b0+2c0+3d0 = b1
#   6) S0''(1)=S1''(1)=> 2c0+6d0 = 2c1
#   7) S0''(0)=0       => 2c0 = 0           (natural)
#   8) S1''(2)=0       => 2c1+6d1 = 0       (natural)

# order: a0 b0 c0 d0 a1 b1 c1 d1
A = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 2, 3, 0,-1, 0, 0],
    [0, 0, 2, 6, 0, 0,-2, 0],
    [0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 2, 6],
])
b = np.array([1, 2, 2, 0, 0, 0, 0, 0])

coeffs = np.linalg.solve(A, b)
a0, b0, c0, d0, a1, b1, c1, d1 = coeffs

print("S0(x) = {} + {}*x + {}*x^2 + {}*x^3".format(a0, b0, c0, d0))
print("S1(x) = {} + {}*(x-1) + {}*(x-1)^2 + {}*(x-1)^3".format(a1, b1, c1, d1))

x0 = np.linspace(0, 1, 200)
x1 = np.linspace(1, 2, 200)
s0 = a0 + b0*x0 + c0*x0**2 + d0*x0**3
s1 = a1 + b1*(x1-1) + c1*(x1-1)**2 + d1*(x1-1)**3

plt.figure(figsize=(8, 5))
plt.plot(x0, s0, 'b-', linewidth=2, label='S0')
plt.plot(x1, s1, 'r-', linewidth=2, label='S1')
plt.plot(x_nodes, y_nodes, 'ko', markersize=8, label='nodes')
plt.title("Natural Cubic Spline")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("outputs/cubic_spline.png", dpi=150)
plt.show()
