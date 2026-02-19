# Final Project: Image Restoration via Numerical Linear Algebra

## Key Idea

A blurred/noisy image can be written as a linear system:

```
b = A·x + noise
```

where `b` is the degraded image, `x` is the unknown clean image, and `A` is a
convolution matrix. Recovering `x` means solving:

```
(AᵀA + λI) x = Aᵀb
```

This is a standard linear system — solved with **Jacobi**, **Gauss-Seidel**, and
direct **Gaussian elimination**. The parameter `λ` controls how much we smooth
vs. how faithful we are to the data.

## Demo

1. Load `astronaut.jpg`
2. Corrupt it (blur + noise)
3. Solve the system with each algorithm
4. Show: original | corrupted | restored side by side
5. Plot convergence (residual vs. iterations) for each solver

## Why It's Interesting

- Real application of everything covered in the course
- Visually compelling results
- Naturally shows *why* condition number matters (removing λ breaks everything)
- Extends the existing Sobel notebook from edge detection → image restoration
