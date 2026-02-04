def f(x):
    return x**4 + 3*x**3 + x**2 - 2*x - 0.5

def bisection(f, a, b, tol=1e-10, max_iter=1000):
    if f(a) * f(b) > 0:
        return None
    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def find_multiple_roots(f, interval_start, interval_end, num_subintervals=1000, tol=1e-10):
    roots = []
    step = (interval_end - interval_start) / num_subintervals
    for i in range(num_subintervals):
        a = interval_start + i * step
        b = a + step
        if f(a) * f(b) < 0:
            root = bisection(f, a, b, tol)
            if root is not None:
                is_duplicate = False
                for r in roots:
                    if abs(r - root) < 1e-6:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    roots.append(root)
    return roots

roots = find_multiple_roots(f, -3, 2)
for root in roots:
    print(f"x = {root}, f(x) = {f(root)}")
