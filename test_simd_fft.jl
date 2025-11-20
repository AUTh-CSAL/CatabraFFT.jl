using CatabraFFT, SIMD, FFTW

println("="^80)
println("Testing SIMD FFT Generation for Small Sizes")
println("="^80)
println()

# Test FFT8
println("Testing FFT8:")
println("-"^80)

x = [ComplexF32(i,i) for i in 1:2^3]
y = similar(x)

println("Input: ", x)
println()

# FFTW reference
F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)
fftw_result = F * x
println("FFTW result: ", fftw_result)
println()

# Try CatabraFFT
try
    C = CatabraFFT.plan_fft(x, CatabraFFT.ENCHANT)
    println("Plan created successfully!")

    x_test = [ComplexF32(i,i) for i in 1:2^3]
    catabra_result = C * x_test

    println("CatabraFFT result: ", catabra_result)
    println()

    if isapprox(catabra_result, fftw_result, rtol=1e-5)
        println("✓ FFT8 PASSED!")
    else
        println("✗ FFT8 FAILED!")
        println("Max error: ", maximum(abs.(catabra_result .- fftw_result)))
    end
catch e
    println("✗ ERROR in FFT8:")
    println(e)
    println()
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println()
println("="^80)

# Test FFT4
println("Testing FFT4:")
println("-"^80)

x4 = [ComplexF32(i,i) for i in 1:4]
println("Input: ", x4)

F4 = FFTW.plan_fft(x4; flags=FFTW.EXHAUSTIVE)
fftw_result4 = F4 * x4
println("FFTW result: ", fftw_result4)
println()

try
    C4 = CatabraFFT.plan_fft(x4, CatabraFFT.ENCHANT)
    println("Plan created successfully!")

    x4_test = [ComplexF32(i,i) for i in 1:4]
    catabra_result4 = C4 * x4_test

    println("CatabraFFT result: ", catabra_result4)
    println()

    if isapprox(catabra_result4, fftw_result4, rtol=1e-5)
        println("✓ FFT4 PASSED!")
    else
        println("✗ FFT4 FAILED!")
        println("Max error: ", maximum(abs.(catabra_result4 .- fftw_result4)))
    end
catch e
    println("✗ ERROR in FFT4:")
    println(e)
    println()
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
