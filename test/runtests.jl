using CatabraFFT
using Test, BenchmarkTools, FFTW, LinearAlgebra, Plots

function plot_complex_signal(data::Vector{ComplexF64}, fftgen_result::Vector{ComplexF64}, fftw_result::Vector{ComplexF64}, title_str::String = "DSP")
    n = length(data)

    real_part = real.(data)
    imag_part = imag.(data)

    real_fftgen = real.(fftgen_result)
    imag_fftgen = imag.(fftgen_result)

    real_fftw = real.(fftw_result)
    imag_fftw = imag.(fftw_result)


    p = plot(layout = (3, 1), size = (800, 600), title=title_str)
    plot!(1:n, real_part, seriestype = :line, title="Real Part", subplot=1)
    plot!(1:n, imag_part, seriestype = :line, title="Imaginary Part", subplot=1)

    plot!(1:n, real_fftw, seriestype = :line, title="Real Part of FFTW", subplot=2)
    plot!(1:n, imag_fftw, seriestype = :line, title="Imaginary Part of FFTW", subplot=2)

    plot!(1:n, real_fftgen, seriestype = :line, title="Real Part of FFTGEN", subplot=3)
    plot!(1:n, imag_fftgen, seriestype = :line, title="Imaginary Part of FFTGEN", subplot=3)
    display(p)
end

function relative_error(x::Vector{ComplexF64}, y::Vector{ComplexF64})
    return norm(x - y) / norm(y)
end

function test(n::Int, plot::Bool=false)
    #x = randn(ComplexF64, n)
    x = ComplexF64[4i + 5im*i for i in 1:n]

    result = CatabraFFT.FFT(x) # run once to compile

    println("Catabra FFT time:")
    @time result = CatabraFFT.FFT(x)
    #@time FFT!(result, x)

    println("FFTW time:")
    @time y = FFTW.fft(x)
    n < 32 && @show result y
    rel_err = relative_error(y, result)
    println("Cached-in functions: $(keys(F_cache))")
    println("Relative Error: $rel_err")
    if plot plot_complex_signal(x, result, y) end
    @assert rel_err < 1e-10
end


function bench(n::Int, fftw_time::Vector, mixed_radix_time::Vector, fftw_mem::Vector, mixed_radix_mem::Vector)
    x = ComplexF64[i + im * i for i in 1:n]

    fftw_result = FFTW.fft(x)
    y_mixed = similar(x)
    FFT!(y_mixed, x)
    @show rel_err = relative_error(y_mixed, fftw_result)
    @assert rel_err < 1e-10


    # Benchmark FFTW
    t_fftw = @benchmark FFTW.fft($x) seconds = 0.01
    push!(fftw_time, log10(median(t_fftw).time / 10^9))
    push!(fftw_mem, log10(median(t_fftw).memory / 1024))

    # Run custom FFT benchmark
    t_mixed = @benchmark FFT!($y_mixed, $x) seconds=0.01
    push!(mixed_radix_time, log10(median(t_mixed).time / 10^9))
    push!(mixed_radix_mem, log10(median(t_mixed).memory / 1024))

end

function benchmark_fft_over_range(xs::Vector)
    fftw_time = []
    mixed_radix_time = []
    fftw_mem = []
    mixed_radix_mem = []
    
    # Precompute all function before benchmarking
    for i in xs
        test(i, false)
    end

    for n in xs
        print("n = $n \n")
        bench(n, fftw_time, mixed_radix_time, fftw_mem, mixed_radix_mem)
        println("time fftw: ", fftw_time[end], " time mixed radix: ", mixed_radix_time[end])
    end

    p_time = plot(log2.(xs), fftw_time, label="FFTW (median)", markershape=:square, markercolor=:red, legend=:topleft)
    plot!(p_time,log2.(xs), mixed_radix_time, label="MIXED-RADIX FFT (median)", markershape=:circle, markercolor=:orange)

    xlabel!(p_time, "log2(Input length)")
    ylabel!(p_time, "log10(Time (sec))")
    title!(p_time, "FFT Performance Comparison: Time")

    display(p_time)

    p_mem = plot(log2.(xs), fftw_mem, label="FFTW (memory)", markershape=:square, markercolor=:red, legend=:topleft)
    plot!(p_mem,log2.(xs), mixed_radix_mem, label="MIXED-RADIX FFT (memory)", markershape=:circle, markercolor=:orange)

    xlabel!(p_mem, "log2(Input length)")
    ylabel!(p_mem, "log10(Memory (KB))")
    title!(p_mem, "FFT Performance Comparison: Memory Allocation")

    #display(p_mem)
end


@testset "CatabraFFT.jl" begin
    n = 3^7
    test(n, false)

    #xs = 2 .^ collect(2:1:27)
    xs = collect(2:1:120)
    #xs = 3 .^ collect(1:1:17)
    #benchmark_fft_over_range(xs)
    println("Done!")

    # Write your tests here.
end
