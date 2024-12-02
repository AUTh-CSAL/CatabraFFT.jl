using CatabraFFT
using BenchmarkTools, FFTW, LinearAlgebra
using Plots

# Bencharks CatabraFFT.jl compared to FFTW.jl plots the results.

relative_error(x, y) = norm(x - y) / norm(y)


function bench(n::Int, fftw_time::Vector, mixed_radix_time::Vector, fftw_mem::Vector, mixed_radix_mem::Vector)
    x = randn(ComplexF64,n)
    F = FFTW.plan_fft(x; flags=FFTW.PATIENT, timelimit=Inf)

    fftw_result = F*x
    # Benchmark FFTW
    t_fftw = @benchmark ($F*$x) 
    push!(fftw_time, log10(median(t_fftw).time / 10^9))
    push!(fftw_mem, log10(median(t_fftw).memory / 1024))

    y_mixed = similar(x)
    FFT!(y_mixed, x)
    @show rel_err = relative_error(y_mixed, fftw_result)
    @assert rel_err < 1e-10

    # Run custom FFT benchmark
    t_mixed = @benchmark FFT!($y_mixed, $x) 
    push!(mixed_radix_time, log10(median(t_mixed).time / 10^9))
    push!(mixed_radix_mem, log10(median(t_mixed).memory / 1024))
end

function benchmark_fft_over_range(xs::Vector)
    fftw_time = []
    mixed_radix_time = []
    fftw_mem = []
    mixed_radix_mem = []

    # Precompute all function before benchmarking
    for n in xs
        CatabraFFT.FFT(rand(ComplexF64, n))
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

    p_mem = plot(log2.(xs), fftw_mem, label="FFTW (memory)", markershape=:square, markercolor=:red, legend=:topleft, reuse=false)
    plot!(p_mem,log2.(xs), mixed_radix_mem, label="MIXED-RADIX FFT (memory)", markershape=:circle, markercolor=:orange)

    xlabel!(p_mem, "log2(Input length)")
    ylabel!(p_mem, "log10(Memory (KB))")
    title!(p_mem, "FFT Performance Comparison: Memory Allocation")

    # display(p_mem)
end

function main()
    xs = 2 .^ (4:27)
    # xs = sort(vcat([2 .^(2:24), 3 .^(2:15), 5 .^(2:10), 7 .^(2:8), 10 .^(2:7)]...))
    benchmark_fft_over_range(xs)
    println("Done!")
end
