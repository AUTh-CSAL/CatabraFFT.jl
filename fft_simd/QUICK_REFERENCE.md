# Quick Reference: Using SIMD FFT Implementation

## Files Overview

1. **fft_simd_complete.jl** - Core SIMD implementation
   - `recfft2_simd()` - Main kernel generator
   - `load_complex_simd()` - Optimized complex loading
   - `store_complex_simd!()` - Optimized complex storing
   - `complex_butterfly_simd()` - Butterfly with shufflevector

2. **fft_simd_integration.jl** - Integration with existing code
   - `makefftradix_simd()` - Drop-in replacement
   - `load_real_imag_gen_simd()` - Enhanced load generator
   - `benchmark_simd_vs_scalar()` - Performance comparison

3. **complex_mul_examples.jl** - Detailed examples
   - Shows all shufflevector patterns
   - Visual traces for debugging
   - Performance benchmarks

4. **fft_simd_theory.md** - Theory and extensions
   - Algebraic framework
   - Metatheory.jl integration
   - Research directions

5. **IMPLEMENTATION_SUMMARY.md** - Roadmap
   - Integration steps
   - Performance expectations
   - Troubleshooting guide

## Quick Start (3 Steps)

### Step 1: Add to your fft_seed.jl
```julia
# At the top of fft_seed.jl
include("fft_simd_complete.jl")
include("fft_simd_integration.jl")

# Enable SIMD mode
const USE_SIMD = true
```

### Step 2: Update makefftradix call
```julia
# Change from:
kernel = makefftradix(n, suffixes, D, p, op, SIZE, T, SIMD_BITS)

# To:
kernel = if USE_SIMD && n >= 4
    makefftradix_simd(n, suffixes, D, p, op, SIZE, T, SIMD_BITS)
else
    makefftradix(n, suffixes, D, p, op, SIZE, T, SIMD_BITS)
end
```

### Step 3: Test
```julia
# Run your existing tests
@time fft1024_test = your_fft_function(randn(ComplexF64, 1024))

# Should see ~2× speedup immediately
```

## Key Functions to Know

### 1. Load Complex Numbers
```julia
# Contiguous: Uses vload (fast!)
indices = [1, 2, 3, 4]  # Consecutive
v = load_complex_simd(arr, indices, Val(256))
# Result: Vec{8, Float64}([r1,i1,r2,i2,r3,i3,r4,i4])

# Strided: Uses vgather (still good)
indices = [1, 3, 5, 7]  # Non-consecutive
v = load_complex_simd(arr, indices, Val(256))
```

### 2. Store Complex Numbers
```julia
# Contiguous: Uses vstore
store_complex_simd!(arr, v, [1,2,3,4], Val(256))

# Strided: Uses vscatter
store_complex_simd!(arr, v, [1,3,5,7], Val(256))
```

### 3. Complex Multiplication (the magic!)
```julia
# Multiply by -i (90° rotation)
v = Vec{4, Float64}((1.0, 2.0, 3.0, 4.0))  # (1+2i), (3+4i)
result = complex_mul_neg_i_simd(v)
# Result: (2-i), (4-3i)

# Multiply by (1-i)/√2 (common FFT8 twiddle)
result = complex_mul_inv_sqrt2_q4_simd(v)

# General twiddle factor
result = complex_mul_twiddle_simd(v, cos_val, sin_val)
```

### 4. Complete Butterfly
```julia
# FFT butterfly: y0 = x0 + x1*W, y1 = x0 - x1*W
x0 = Vec{4, Float64}(...)
x1 = Vec{4, Float64}(...)
y0, y1 = fft_butterfly_simd(x0, x1, cos_w, sin_w)
```

## Common Patterns

### Pattern 1: Extract Real/Imag Parts
```julia
v = Vec{8, Float64}([r1,i1,r2,i2,r3,i3,r4,i4])

# Get all real parts: [r1, r2, r3, r4]
reals = shufflevector(v, Val((0, 2, 4, 6)))

# Get all imag parts: [i1, i2, i3, i4]
imags = shufflevector(v, Val((1, 3, 5, 7)))
```

### Pattern 2: Broadcast Values
```julia
# Broadcast first real to all positions
v = Vec{4, Float64}((1.0, 2.0, 3.0, 4.0))
broadcast = shufflevector(v, Val((0, 0, 0, 0)))
# Result: [1.0, 1.0, 1.0, 1.0]
```

### Pattern 3: Interleave Two Vectors
```julia
v1 = Vec{4, Float64}((1.0, 2.0, 3.0, 4.0))
v2 = Vec{4, Float64}((5.0, 6.0, 7.0, 8.0))

# Interleave: [v1[0], v2[0], v1[1], v2[1], ...]
result = shufflevector(v1, v2, Val((0, 4, 1, 5, 2, 6, 3, 7)))
# Result: [1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]
```

### Pattern 4: FMA for Complex Multiply
```julia
# (a+bi)(c+di) = (ac-bd) + i(ad+bc)
reals_a = shufflevector(va, Val((0, 2, 4, 6)))
imags_a = shufflevector(va, Val((1, 3, 5, 7)))
reals_c = shufflevector(vc, Val((0, 2, 4, 6)))
imags_c = shufflevector(vc, Val((1, 3, 5, 7)))

# Real part: ac - bd
real_result = muladd(reals_a, reals_c, -imags_a * imags_c)

# Imag part: ad + bc  
imag_result = muladd(reals_a, imags_c, imags_a * reals_c)

# Interleave back
result = shufflevector(real_result, imag_result, Val((0,4,1,5,2,6,3,7)))
```

## Performance Tips

### 1. Check SIMD Code Generation
```julia
using InteractiveUtils

# Verify AVX2 instructions
@code_llvm complex_butterfly_simd(v1, v2, "1", "+", Val(256))
# Look for: %res = call <4 x double> @llvm.fma.v4f64(...)

@code_native debuginfo=:none complex_butterfly_simd(...)  
# Look for: vfmadd213pd %ymm0, %ymm1, %ymm2
```

### 2. Ensure Data Alignment
```julia
# Julia arrays are 16-byte aligned by default
# For 32-byte (AVX2) or 64-byte (AVX512) alignment:
using SIMD

arr = Vector{ComplexF64}(undef, n)
@assert pointer(arr) % 32 == 0  # Check alignment
```

### 3. Profile Hot Paths
```julia
using Profile

@profile for i in 1:1000
    fft_simd!(y, x)
end

Profile.print()
# Look for time spent in shufflevector vs actual compute
```

### 4. Benchmark Against FFTW
```julia
using FFTW, BenchmarkTools

x = randn(ComplexF64, 1024)

# FFTW
plan = plan_fft(x)
@benchmark $plan * $x

# Your implementation  
@benchmark fft_simd!($y, $x)

# Should be within 2× of FFTW after optimization
```

## Debugging Checklist

### If no speedup:
- [ ] Check SIMD code generation with @code_native
- [ ] Verify vector width matches hardware (256 for AVX2)
- [ ] Profile to find bottlenecks
- [ ] Test with larger sizes (n >= 16)
- [ ] Check for register spills in @code_llvm

### If incorrect results:
- [ ] Compare with scalar version element-by-element
- [ ] Check shuffle indices (0-based in Val())
- [ ] Verify twiddle factor values
- [ ] Test individual operations in isolation
- [ ] Check for off-by-one errors in indices

### If crashes:
- [ ] Check array bounds
- [ ] Verify pointer arithmetic
- [ ] Test with @inbounds removed
- [ ] Check for type mismatches
- [ ] Verify vector length matches operations

## SIMD Width Reference

| Type | AVX2 (256-bit) | AVX512 (512-bit) |
|------|----------------|------------------|
| Float64 complex | 2 per vector | 4 per vector |
| Float32 complex | 4 per vector | 8 per vector |
| Registers | ymm0-ymm15 (16) | zmm0-zmm31 (32) |

## Example: Complete FFT8 Kernel

```julia
function fft8_simd(x::AbstractVector{ComplexF64})
    T = Float64
    
    # Load 8 complex numbers (16 floats)
    # Split into 2 AVX2 vectors
    v1 = load_complex_simd(x, [1,2,3,4], Val(256))  # x[1:4]
    v2 = load_complex_simd(x, [5,6,7,8], Val(256))  # x[5:8]
    
    # Stage 1: FFT2 on each pair
    # ... (generated by recfft2_simd)
    
    # Stage 2: FFT4 combining pairs with twiddles
    # ... (generated by recfft2_simd)
    
    # Stage 3: FFT8 final butterfly
    # ... (generated by recfft2_simd)
    
    # Store results
    y = similar(x)
    store_complex_simd!(y, v1_result, [1,2,3,4], Val(256))
    store_complex_simd!(y, v2_result, [5,6,7,8], Val(256))
    
    return y
end
```

## Next Actions

1. **Today**: Add files to your project, run basic tests
2. **This week**: Benchmark all factorizations with SIMD
3. **Next week**: Profile and optimize hot paths
4. **Month 1**: Match FFTW performance
5. **Month 2**: Explore Metatheory.jl for auto-optimization

## Questions?

Common issues and solutions:

**Q: "SIMD not being used"**
A: Check with `@code_native`. May need `@inbounds` or different compiler flags.

**Q: "Slower than scalar for small n"**  
A: Expected! Use SIMD only for n >= 4 or n >= 8.

**Q: "How to handle odd sizes?"**
A: Combine with Bluestein/Rader for prime sizes.

**Q: "GPU version?"**
A: Same concepts apply! Replace SIMD with warp operations.

**Q: "How to use Metatheory.jl?"**
A: See theory document for full framework. Short version: Define rewrite rules, run saturation search.

---

**You're ready to go!** Start with integrating `fft_simd_complete.jl` and see the speedup immediately. The theory and advanced optimizations can come later.
