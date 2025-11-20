using CatabraFFT

# Test recfft2_simd code generation
include("src/fft_seed.jl")

println("="^80)
println("Testing recfft2_simd code generation")
println("="^80)
println()

# Test FFT4 SIMD
println("FFT4 SIMD (Float32, AVX2 256-bit):")
println("-"^80)
y4 = ["y$i" for i in 1:4]
x4 = ["x$i" for i in 1:4]
code4 = recfft2_simd(y4, x4, nothing, nothing, true, Float32; SIMD_WIDTH=256, tmp_base=1, mode=:default, py="")
println(code4)
println()

# Test FFT8 SIMD
println("="^80)
println("FFT8 SIMD (Float32, AVX2 256-bit):")
println("-"^80)
y8 = ["y$i" for i in 1:8]
x8 = ["x$i" for i in 1:8]
code8 = recfft2_simd(y8, x8, nothing, nothing, true, Float32; SIMD_WIDTH=256, tmp_base=1, mode=:default, py="")
println(code8)
println()

# Test FFT4 scalar for comparison
println("="^80)
println("FFT4 Scalar (for comparison):")
println("-"^80)
code4_scalar = recfft2(y4, x4, nothing, nothing, true, Float32, 1, :default, "")
println(code4_scalar)
println()
