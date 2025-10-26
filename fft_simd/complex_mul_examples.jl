# Complex Multiplication with shufflevector: Detailed Examples

"""
This file demonstrates how to use shufflevector operations for efficient
complex number arithmetic in SIMD FFT kernels. Each example shows:
1. The mathematical operation
2. The naive approach
3. The optimized shufflevector approach
4. Register contents at each step
"""

using SIMD, BenchmarkTools

# ============================================================================
# Example 1: Basic Complex Multiplication
# ============================================================================

"""
Multiply two complex numbers using shufflevector:
(a + bi) * (c + di) = (ac - bd) + i(ad + bc)

Input:  v1 = Vec{4, Float64}([a, b, c_unused, d_unused])
        v2 = Vec{4, Float64}([c, d, e_unused, f_unused])
Output: result = Vec{4, Float64}([ac-bd, ad+bc, ..., ...])
"""
function complex_mul_single_simd(v1::Vec{4, T}, v2::Vec{4, T}) where T
    # Method 1: Naive approach (many operations)
    # a, b = v1[0], v1[1]
    # c, d = v2[0], v2[1]
    # real = a*c - b*d
    # imag = a*d + b*c
    
    # Method 2: Optimized with shufflevector
    # Step 1: Broadcast a and b across lanes
    a_broadcast = shufflevector(v1, Val((0, 0)))  # [a, a]
    b_broadcast = shufflevector(v1, Val((1, 1)))  # [b, b]
    
    # Step 2: Arrange c and d for parallel operations
    cd = shufflevector(v2, Val((0, 1)))           # [c, d]
    dc = shufflevector(v2, Val((1, 0)))           # [d, c]
    
    # Step 3: Parallel multiply and combine
    # real = a*c - b*d
    # imag = a*d + b*c
    ac_ad = a_broadcast * cd                       # [a*c, a*d]
    bd_bc = b_broadcast * dc                       # [b*d, b*c]
    
    # Step 4: Final combination with sign adjustment
    signs = Vec{2, T}((1, 1))
    result = muladd(signs, ac_ad, Vec{2,T}((-1, 1)) * bd_bc)
    
    return result
end

# Visual trace:
"""
Input registers:
  v1 = [1.0, 2.0, ?, ?]     # (1 + 2i)
  v2 = [3.0, 4.0, ?, ?]     # (3 + 4i)

After broadcast:
  a_broadcast = [1.0, 1.0]
  b_broadcast = [2.0, 2.0]
  cd = [3.0, 4.0]
  dc = [4.0, 3.0]

After multiply:
  ac_ad = [3.0, 4.0]        # [a*c, a*d]
  bd_bc = [8.0, 6.0]        # [b*d, b*c]

After FMA with signs:
  result = [3.0 - 8.0, 4.0 + 6.0] = [-5.0, 10.0]

Verification: (1+2i)(3+4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i ✓
"""

# ============================================================================
# Example 2: Vectorized Complex Multiplication (2 pairs)
# ============================================================================

"""
Multiply two pairs of complex numbers simultaneously:
(a₁ + b₁i) * (c₁ + d₁i) and (a₂ + b₂i) * (c₂ + d₂i)

Input layout (interleaved):
  v1 = [a₁, b₁, a₂, b₂]  # Two complex numbers
  v2 = [c₁, d₁, c₂, d₂]  # Two complex numbers
  
Output: [r₁, i₁, r₂, i₂]  # Two results
"""
function complex_mul_vec2_simd(v1::Vec{N, T}, v2::Vec{N, T}) where {N, T}
    # Extract real and imaginary parts using shufflevector
    # Real parts at even indices: 0, 2
    # Imag parts at odd indices: 1, 3
    
    reals_v1 = shufflevector(v1, Val((0, 2)))     # [a₁, a₂]
    imags_v1 = shufflevector(v1, Val((1, 3)))     # [b₁, b₂]
    
    reals_v2 = shufflevector(v2, Val((0, 2)))     # [c₁, c₂]
    imags_v2 = shufflevector(v2, Val((1, 3)))     # [d₁, d₂]
    
    # Parallel complex multiplication:
    # real_result = real₁ * real₂ - imag₁ * imag₂
    # imag_result = real₁ * imag₂ + imag₁ * real₂
    
    # Use FMA for efficiency
    real_result = muladd(reals_v1, reals_v2, -imags_v1 * imags_v2)
    imag_result = muladd(reals_v1, imags_v2,  imags_v1 * reals_v2)
    
    # Interleave results back: [r₁, i₁, r₂, i₂]
    # Real results go to indices 0, 2; imag to 1, 3
    result = shufflevector(real_result, imag_result, Val((0, 2, 1, 3)))
    
    return result
end

# Visual trace:
"""
Input:
  v1 = [1.0, 2.0, 3.0, 4.0]    # (1+2i), (3+4i)
  v2 = [5.0, 6.0, 7.0, 8.0]    # (5+6i), (7+8i)

After extraction:
  reals_v1 = [1.0, 3.0]
  imags_v1 = [2.0, 4.0]
  reals_v2 = [5.0, 7.0]
  imags_v2 = [6.0, 8.0]

After multiplication:
  reals_v1 * reals_v2 = [5.0, 21.0]
  imags_v1 * imags_v2 = [12.0, 32.0]
  reals_v1 * imags_v2 = [6.0, 24.0]
  imags_v1 * reals_v2 = [10.0, 28.0]

After FMA:
  real_result = [5.0 - 12.0, 21.0 - 32.0] = [-7.0, -11.0]
  imag_result = [6.0 + 10.0, 24.0 + 28.0] = [16.0, 52.0]

After interleave:
  result = [-7.0, 16.0, -11.0, 52.0]

Verification:
  (1+2i)(5+6i) = 5 + 6i + 10i + 12i² = -7 + 16i ✓
  (3+4i)(7+8i) = 21 + 24i + 28i + 32i² = -11 + 52i ✓
"""

# ============================================================================
# Example 3: Multiply by -i (90° rotation)
# ============================================================================

"""
Multiply complex number by -i (rotate by -90°):
(a + bi) * (-i) = -ai - bi² = b - ai

This is just a shuffle + sign change!
"""
function complex_mul_neg_i_simd(v::Vec{N, T}) where {N, T}
    n_complex = N ÷ 2
    
    # Swap real and imaginary parts, negate new real
    # [r₁, i₁, r₂, i₂, ...] → [i₁, -r₁, i₂, -r₂, ...]
    
    # Extract parts
    reals = shufflevector(v, Val(Tuple(2*i for i in 0:n_complex-1)))
    imags = shufflevector(v, Val(Tuple(2*i+1 for i in 0:n_complex-1)))
    
    # Interleave with sign change: [i, -r, i, -r, ...]
    result = shufflevector(imags, -reals, 
                          Val(Tuple(vcat([[i, i+n_complex] for i in 0:n_complex-1]...))))
    
    return result
end

# Example for N=4 (2 complex numbers):
"""
Input:  v = [1.0, 2.0, 3.0, 4.0]    # (1+2i), (3+4i)

After shuffle:
  reals = [1.0, 3.0]
  imags = [2.0, 4.0]
  -reals = [-1.0, -3.0]

After interleave:
  result = [2.0, -1.0, 4.0, -3.0]   # (2-1i), (4-3i)

Verification:
  (1+2i)(-i) = -i - 2i² = 2 - i ✓
  (3+4i)(-i) = -3i - 4i² = 4 - 3i ✓
"""

# ============================================================================
# Example 4: Multiply by (1-i)/√2 (FFT8 twiddle)
# ============================================================================

"""
Multiply by (1-i)/√2 - common twiddle factor in FFT8:
(a + bi) * (1-i)/√2 = [(a+b) + i(b-a)]/√2

Strategy:
1. Compute (a+b) and (b-a) in parallel
2. Multiply by 1/√2
3. Interleave
"""
function complex_mul_inv_sqrt2_q4_simd(v::Vec{N, T}) where {N, T}
    n_complex = N ÷ 2
    
    # Extract real and imaginary parts
    real_indices = Tuple(2*i for i in 0:n_complex-1)
    imag_indices = Tuple(2*i+1 for i in 0:n_complex-1)
    
    reals = shufflevector(v, Val(real_indices))
    imags = shufflevector(v, Val(imag_indices))
    
    # Compute sums and differences
    sum_ri = (reals + imags) * T(0.7071067811865476)  # (a+b)/√2
    diff_ir = (imags - reals) * T(0.7071067811865476)  # (b-a)/√2
    
    # Interleave: [sum₁, diff₁, sum₂, diff₂, ...]
    interleave_pattern = Tuple(vcat([[i, i+n_complex] for i in 0:n_complex-1]...))
    result = shufflevector(sum_ri, diff_ir, Val(interleave_pattern))
    
    return result
end

# Visual trace:
"""
Input:  v = [2.0, 2.0, 4.0, 0.0]    # (2+2i), (4+0i)

After extraction:
  reals = [2.0, 4.0]
  imags = [2.0, 0.0]

After operations:
  sum_ri = [4.0, 4.0] * 0.707... = [2.828..., 2.828...]
  diff_ir = [0.0, -4.0] * 0.707... = [0.0, -2.828...]

After interleave:
  result = [2.828..., 0.0, 2.828..., -2.828...]

Verification:
  (2+2i)(1-i)/√2 = (2-2i+2i-2i²)/√2 = 4/√2 = 2√2 ≈ 2.828 ✓
  (4+0i)(1-i)/√2 = (4-4i)/√2 = 2√2 - 2√2i ✓
"""

# ============================================================================
# Example 5: General Twiddle Factor with CISPI
# ============================================================================

"""
Multiply by general twiddle factor: cos(θ) + i*sin(θ)

This shows the full pattern for arbitrary rotations.
"""
function complex_mul_twiddle_simd(v::Vec{N, T}, cos_val::T, sin_val::T) where {N, T}
    n_complex = N ÷ 2
    
    # Extract real and imaginary parts
    reals = shufflevector(v, Val(Tuple(2*i for i in 0:n_complex-1)))
    imags = shufflevector(v, Val(Tuple(2*i+1 for i in 0:n_complex-1)))
    
    # Broadcast twiddle values
    cos_vec = Vec{n_complex, T}(ntuple(_ -> cos_val, n_complex))
    sin_vec = Vec{n_complex, T}(ntuple(_ -> sin_val, n_complex))
    
    # Complex multiplication using FMA:
    # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
    # real_result = a*cos - b*sin
    # imag_result = a*sin + b*cos
    
    real_result = muladd(reals, cos_vec, -imags * sin_vec)
    imag_result = muladd(reals, sin_vec,  imags * cos_vec)
    
    # Interleave back
    interleave = Tuple(vcat([[i, i+n_complex] for i in 0:n_complex-1]...))
    result = shufflevector(real_result, imag_result, Val(interleave))
    
    return result
end

# Example with θ = π/4 (45° rotation):
"""
Input:  v = [1.0, 0.0, 0.0, 1.0]    # (1+0i), (0+1i)
Twiddle: cos(π/4) + i*sin(π/4) = 0.707... + 0.707...i

After extraction:
  reals = [1.0, 0.0]
  imags = [0.0, 1.0]
  cos_vec = [0.707..., 0.707...]
  sin_vec = [0.707..., 0.707...]

After FMA:
  real_result = [1.0*0.707 - 0*0.707, 0*0.707 - 1.0*0.707]
              = [0.707..., -0.707...]
  imag_result = [1.0*0.707 + 0*0.707, 0*0.707 + 1.0*0.707]
              = [0.707..., 0.707...]

Result: [0.707..., 0.707..., -0.707..., 0.707...]
      = (0.707+0.707i), (-0.707+0.707i)

Verification:
  1 * e^(iπ/4) = cos(π/4) + i*sin(π/4) ✓
  i * e^(iπ/4) = i(cos(π/4) + i*sin(π/4)) = i*cos - sin = -sin + i*cos ✓
"""

# ============================================================================
# Example 6: Full FFT Butterfly with Twiddle
# ============================================================================

"""
Complete FFT butterfly operation:
y₀ = x₀ + x₁ * W
y₁ = x₀ - x₁ * W

Where W is a twiddle factor. Shows full register usage pattern.
"""
function fft_butterfly_simd(x0::Vec{N, T}, x1::Vec{N, T}, 
                           cos_w::T, sin_w::T) where {N, T}
    # Step 1: Apply twiddle to x1
    x1_twisted = complex_mul_twiddle_simd(x1, cos_w, sin_w)
    
    # Step 2: Butterfly - just add/subtract!
    y0 = x0 + x1_twisted
    y1 = x0 - x1_twisted
    
    return (y0, y1)
end

# Visual trace for 2 complex numbers:
"""
Input:
  x0 = [1.0, 0.0, 2.0, 0.0]    # (1+0i), (2+0i)
  x1 = [1.0, 0.0, 0.0, 2.0]    # (1+0i), (0+2i)
  W = 1 (no rotation)

After twiddle (W=1, so unchanged):
  x1_twisted = [1.0, 0.0, 0.0, 2.0]

After butterfly:
  y0 = [2.0, 0.0, 2.0, 2.0]    # (2+0i), (2+2i)
  y1 = [0.0, 0.0, 2.0, -2.0]   # (0+0i), (2-2i)

This is a radix-2 DIT FFT step!
"""

# ============================================================================
# Performance Comparison
# ============================================================================

"""
Compare performance of scalar vs SIMD complex multiplication.
"""
function benchmark_complex_mul()
    
    N = 1024
    
    # Scalar version
    a_scalar = [1.0 + 2.0im for _ in 1:N]
    b_scalar = [3.0 + 4.0im for _ in 1:N]
    c_scalar = similar(a_scalar)
    
    println("Scalar complex multiplication:")
    @btime begin
        for i in 1:$N
            $c_scalar[i] = $a_scalar[i] * $b_scalar[i]
        end
    end
    
    # SIMD version (process 4 at a time with Float64)
    # Each Vec{8, Float64} holds 4 complex numbers
    println("\nSIMD complex multiplication (4 pairs at a time):")
    a_simd = reinterpret(Float64, a_scalar)
    b_simd = reinterpret(Float64, b_scalar)
    c_simd = similar(a_simd)
    
    @btime begin
        for i in 1:8:length($a_simd)
            v1 = vload(Vec{8, Float64}, $a_simd, i)
            v2 = vload(Vec{8, Float64}, $b_simd, i)
            result = complex_mul_vec2_simd(v1, v2)
            vstore(result, $c_simd, i)
        end
    end
    
    println("\nExpected speedup: ~2-4× (depending on hardware)")
end

# ============================================================================
# Register Allocation Strategy
# ============================================================================

"""
Guidelines for managing ymm/zmm registers efficiently:

AVX2 (256-bit, 16 registers):
- ymm0-ymm3: Input data (4 registers)
- ymm4-ymm7: Intermediate results (4 registers)
- ymm8-ymm11: Twiddle factors (4 registers)
- ymm12-ymm15: Temporaries (4 registers)

AVX512 (512-bit, 32 registers):
- zmm0-zmm7: Input data (8 registers)
- zmm8-zmm15: Intermediate results (8 registers)
- zmm16-zmm23: Twiddle factors (8 registers)
- zmm24-zmm31: Temporaries (8 registers)

Strategy:
1. Preload twiddle factors into dedicated registers
2. Use separate registers for real/imag during computation
3. Reuse registers after values are consumed
4. Minimize register spills by careful scheduling
"""

println("Complex multiplication examples loaded.")
println("Run benchmark_complex_mul() to see performance comparison.")
benchmark_complex_mul()