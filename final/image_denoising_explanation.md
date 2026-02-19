# Image Denoising — The Full Explanation

---

## Part 1: The Problem

A noisy image has random "salt and pepper" values sprinkled on every pixel. Each pixel was corrupted independently:

```
corrupted pixel = true pixel + random noise
```

Mathematically:
```
b = x + η
```

| Symbol | Meaning |
|--------|---------|
| `x` | the unknown clean image we want |
| `b` | the noisy image we have |
| `η` | random noise on every pixel |

**The question:** given only `b`, can we recover `x`?

---

## Part 2: Why You Can't Just Subtract the Noise

The noise `η` is completely random. We don't know what it is for each pixel — that's the whole problem. We can't just do `x = b - η` because we don't have `η`.

So we need a smarter approach. We need to use what we **do** know about real images.

---

## Part 3: The Key Insight — Real Images Are Smooth

Look at any natural photo. Neighboring pixels are almost always similar in value. Edges exist, but they're rare. Most of the image is gradual, smooth transitions.

```
Real image:           Noisy image:
pixel values          pixel values
0.5 0.5 0.6 0.6       0.5 0.8 0.3 0.6
0.5 0.6 0.6 0.7  →    0.2 0.6 0.9 0.4
0.6 0.6 0.7 0.7       0.6 0.3 0.7 0.1
(smooth, gradual)     (jumpy, random)
```

Noise **breaks smoothness**. So to remove noise, we find an image that is:

1. **Close to the noisy input** (don't change pixels too much)
2. **As smooth as possible** (neighboring pixels should be similar)

These two goals fight each other. We balance them with a parameter **λ**.

---

## Part 4: The Math — One Objective Function

Write both goals as a single expression to minimize:

```
minimize:    ||x - b||²    +    λ · ||∇x||²
                ↑                      ↑
          "stay close              "be smooth"
           to the data"
```

**Term 1** — `||x - b||²` is the sum of squared differences between restored and noisy:
```
||x - b||² = (x₁-b₁)² + (x₂-b₂)² + (x₃-b₃)² + ...   (one term per pixel)
```

**Term 2** — `||∇x||²` is the sum of squared differences between neighboring pixels:
```
||∇x||² = (x₂-x₁)² + (x₃-x₂)² + ...   horizontal neighbors
         + (x below - x above)² + ...   vertical neighbors
```

If neighbors have similar values → small differences → small penalty.
If neighbors jump wildly (noise) → large differences → large penalty.

**λ controls the balance:**
- λ small → term 2 barely matters → keep the noisy data as-is
- λ large → term 2 dominates → force everything smooth (may lose detail)
- λ = 3 → good middle ground

---

## Part 5: Turn "Minimize" Into a Linear System

To find the minimum, take the **derivative** with respect to `x` and set it to zero:

```
d/dx [ ||x - b||² + λ||∇x||² ] = 0

2(x - b) + 2λ · L · x = 0

x - b + λLx = 0

x + λLx = b

(I + λL) · x = b
```

That's it — a standard linear system. We rename `I + λL` as `M`:

```
M · x = b
```

Solve for `x` and you have the restored image.

---

## Part 6: What Is L — The Laplacian Matrix

`L` encodes the smoothness rule. For every pixel, it computes: **this pixel minus the average of its neighbors**.

**Building it in 1D first — 4 pixels in a row:**

The differences between neighbors are:
```
x₂ - x₁
x₃ - x₂
x₄ - x₃
```

The matrix `Dh` that computes these differences:

```
Dh = [-1   1   0   0]
     [ 0  -1   1   0]
     [ 0   0  -1   1]
```

Check: `Dh · [x₁, x₂, x₃, x₄]ᵀ = [x₂-x₁, x₃-x₂, x₄-x₃]ᵀ` ✓

Then `L = DhᵀDh`:

```
      = [ 1  -1   0   0]
        [-1   2  -1   0]
        [ 0  -1   2  -1]
        [ 0   0  -1   1]
```

**Reading each row of L:**
- Row 1: `x₁ - x₂` → pixel 1 minus its right neighbor
- Row 2: `-x₁ + 2x₂ - x₃` → pixel 2 minus both neighbors (classic Laplacian)
- Row 3: `-x₂ + 2x₃ - x₄` → same pattern
- Row 4: `-x₃ + x₄` → pixel 4 minus its left neighbor

If `L·x = 0` everywhere, the image is perfectly smooth. The larger `L·x` is, the noisier the image.

**For 2D:** we do this for horizontal AND vertical directions, then add:
```
L = DhᵀDh + DvᵀDv
```

`kron` (Kronecker product) extends the 1D version across all rows and columns at once.

---

## Part 7: Full Worked Example — 4 Pixels in a Row

Say we have 4 noisy pixels:
```
b = [0.3,  0.9,  0.2,  0.8]   ← very jumpy (noisy)
```

With λ = 1, build M = I + L:

```
M = I + L = [ 2  -1   0   0]
            [-1   3  -1   0]
            [ 0  -1   3  -1]
            [ 0   0  -1   2]
```

Solve `M·x = b` using Gaussian Elimination:

**Forward elimination on [M | b]:**

```
[ 2  -1   0   0 | 0.3]
[-1   3  -1   0 | 0.9]    R2 = R2 + (1/2)R1
[ 0  -1   3  -1 | 0.2]
[ 0   0  -1   2 | 0.8]

→ [ 2  -1    0    0  | 0.30]
  [ 0  2.5  -1    0  | 1.05]    R3 = R3 + (2/5)R2
  [ 0  -1    3   -1  | 0.20]
  [ 0   0   -1    2  | 0.80]

→ [ 2  -1    0    0  | 0.30]
  [ 0  2.5  -1    0  | 1.05]
  [ 0   0   2.6  -1  | 0.62]    R4 = R4 - (2/5.2)R3
  [ 0   0   -1    2  | 0.80]

→ [ 2  -1    0    0  | 0.30]
  [ 0  2.5  -1    0  | 1.05]
  [ 0   0   2.6  -1  | 0.62]
  [ 0   0    0   1.62| 1.04]
```

**Back substitution:**
```
x₄ = 1.04 / 1.62        = 0.64
x₃ = (0.62 + x₄) / 2.6  = (0.62 + 0.64) / 2.6 = 0.48
x₂ = (1.05 + x₃) / 2.5  = (1.05 + 0.48) / 2.5 = 0.61
x₁ = (0.30 + x₂) / 2    = (0.30 + 0.61) / 2   = 0.46
```

**Result:**
```
Noisy:     b = [0.30,  0.90,  0.20,  0.80]   ← jumps wildly
Restored:  x = [0.46,  0.61,  0.48,  0.64]   ← smooth and consistent
```

The wild spikes (0.9, 0.8) were pulled down. The low dip (0.2) was pulled up. The result is smooth — noise removed.

---

## Part 8: Why Sparse Matrices

For a 128×128 image: `n = 16,384 pixels`. Matrix M is 16,384 × 16,384.

| Storage | Size |
|---------|------|
| Dense   | 16,384² × 8 bytes = **2 GB** |
| Sparse  | ~4 values per row × 16,384 = **~500 KB** |

Why so sparse? Because L only connects each pixel to its 4 neighbors. Every row of M has at most 5 non-zero values. Everything else is zero.

```
Interior pixel row of M:
[0, 0, ..., -λ, ..., -λ, (1+4λ), -λ, ..., -λ, ..., 0]
                ↑          ↑        ↑       ↑
             left       above    center   right
            neighbor   neighbor           neighbor
```

`scipy.sparse` stores only those 5 values per row — not the 16,379 zeros.

---

## Part 9: The Three Solvers

All three solve the same system `M·x = b`. They just go about it differently.

### Jacobi

Start with a guess (all pixels = noisy values). Each iteration, update every pixel simultaneously using the current neighbours:

```
x_i(new) = [ b_i  +  λ·(left + right + above + below) ]  /  (1 + 4λ)
```

Each pixel moves toward the weighted average of itself and its neighbours. Uses **old** values for all neighbours every iteration.

```
Iteration 0:   noisy image
Iteration 50:  slightly smoother
Iteration 175: converged ✓
```

### Gauss-Seidel

Same idea, but uses **freshly updated values immediately** as it goes pixel by pixel:

```
x₁(new) = update using old x₂, x₃ ...
x₂(new) = update using NEW x₁, old x₃ ...   ← uses fresh x₁ immediately
x₃(new) = update using NEW x₁, NEW x₂ ...
```

Because it uses fresher information each step, it converges in roughly **half the iterations** of Jacobi.

```
Jacobi:        175 iterations
Gauss-Seidel:   69 iterations   ← 2.5× faster
```

### Direct (Gaussian Elimination)

Doesn't iterate at all. Factorises M into triangular form and solves exactly in one operation — as shown in the worked example above, but for 16,384 equations at once.

```
Residual:
  Jacobi        9.89 × 10⁻⁶
  Gauss-Seidel  8.99 × 10⁻⁶
  Direct        1.34 × 10⁻¹³  ← machine precision, essentially exact
```

---

## The Complete Picture

```
Noisy image  b  (128×128)
      │
      │  Build L  (sparse Laplacian — encodes smoothness)
      │  Build M = I + λL  (system matrix, 16384×16384 sparse)
      │
      │  Solve  M · x = b
      │
      ├─ Jacobi         → 175 iterations  → restored image
      ├─ Gauss-Seidel   →  69 iterations  → restored image
      └─ Direct (GE)    →   1 exact solve → restored image
```

**One sentence summary:** find the smoothest image that still resembles what we observed — and that is a linear system solvable with the exact same tools from the course.

---

## Effect of λ

| λ | Result |
|---|--------|
| 0.1 | barely smoothed, still noisy |
| 1.0 | slightly cleaner |
| 3.0 | clearly denoised ← used in demo |
| 10.0 | over-smoothed, detail lost |
