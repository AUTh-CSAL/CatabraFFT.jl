using CatabraFFT
using BenchmarkTools, FFTW, LinearAlgebra
using Plots

plotlyjs()

# Bencharks CatabraFFT.jl compared to FFTW.jl plots the results.

relative_error(x, y) = norm(x - y) / norm(y)

function bench(n::Int, fftw_time::Vector, mixed_radix_time::Vector, fftw_mem::Vector, mixed_radix_mem::Vector; ctype=ComplexF64, plan_type=FFTW.MEASURE)

    F = FFTW.plan_fft(randn(ctype, n); flags=plan_type, timelimit=Inf)
    x = randn(ctype, n)

    fftw_result = F * x
    y_mixed = similar(x)
    CatabraFFT.fft!(y_mixed, x) 
    @show rel_err = relative_error(y_mixed, fftw_result)
    @assert y_mixed ≈ fftw_result

    # Benchmark FFTW
    t_fftw = @benchmark $F * x setup=(x = randn($ctype,$n))
    push!(fftw_time, (median(t_fftw).time / 10^9))
    push!(fftw_mem, (median(t_fftw).memory / 1024))

    # Run custom FFT benchmark
    t_mixed = @benchmark CatabraFFT.fft!($y_mixed, x) setup=(x = randn($ctype,$n))
    push!(mixed_radix_time, (median(t_mixed).time / 10^9))
    push!(mixed_radix_mem, (median(t_mixed).memory / 1024))

end

function benchmark_fft_over_range(xs::Vector; ctype=ComplexF64, plan_type=FFTW.MEASURE)
    gflops_catabra = []
    gflops_fftw = []
    fftw_time = []
    mixed_radix_time = []
    fftw_mem = []
    mixed_radix_mem = []

    # Precompute all function before benchmarking
    for n in xs
        CatabraFFT.fft(rand(ctype, n))
    end

    for n in xs
        print("n = $n \n")
        bench(n, fftw_time, mixed_radix_time, fftw_mem, mixed_radix_mem; ctype, plan_type)
        println("time fftw: ", fftw_time[end], " Time mixed radix: ", mixed_radix_time[end])

        push!(gflops_catabra, (5 * n * log2(n) * 10^(-9)) / mixed_radix_time[end])
        push!(gflops_fftw, (5 * n * log2(n) * 10^(-9)) / fftw_time[end])
    end

    info = Sys.cpu_info()[1]
    cpu = "$(info.model)@$(info.speed)GHz"

    p_reltime = bar(
        log2.(xs), fftw_time ./ mixed_radix_time, label="", 
        linestyle=:none, markershape=:square, markercolor=:red, legend=:bottom)
    
    xlabel!(p_reltime, "log2(n)")
    ylabel!(p_reltime, "Relative Time (FFTW / CatabraFFT)")
    title!(p_reltime, "$ctype FFT: Speedup ($cpu)")

    display(p_reltime)

    p_time = plot(
        log2.(xs), log10.(fftw_time), label="$plan_type (median)", 
        linestyle=:solid, markershape=:square, markercolor=:red, legend=:bottomright)
    plot!(p_time,
        log2.(xs), log10.(mixed_radix_time), label="CatabraFFT (median)",
        linestyle=:solid, markershape=:circle, markercolor=:orange)

    xlabel!(p_time, "log2(n)")
    ylabel!(p_time, "log10(Time (sec))")
    title!(p_time, "$ctype FFT: Time ($cpu)")

    display(p_time)

    p_gflops = plot(
        log2.(xs), gflops_fftw, label="$plan_type GFLOPS (median)",
        linestyle=:solid, markershape=:square, markercolor=:red, legend=:bottom)
    plot!(p_gflops,
        log2.(xs), gflops_catabra, label="CatabraFFT GFLOPS (median)",
        linestyle=:solid, markershape=:circle, markercolor=:orange)

    xlabel!(p_gflops, "log2(Input length)")
    ylabel!(p_gflops, "GFLOPS")
    title!(p_gflops, "$ctype FFT: GFLOPS ($cpu)")

    display(p_gflops)

    # p_mem = plot(
    #     log2.(xs), log10.(fftw_mem), label="FFTW (memory)",
    #     linestyle=:none, markershape=:square, markercolor=:red, legend=:topleft)
    # plot!(p_mem,
    #     log2.(xs), log10.(mixed_radix_mem), label="CatabraFFT (memory)",
    #     linestyle=:none, markershape=:circle, markercolor=:orange)

    # xlabel!(p_mem, "log2(Input length)")
    # ylabel!(p_mem, "log10(Memory (KB))")
    # title!(p_mem, "FFT Performance Comparison: Memory Allocation")

    # display(p_mem)
end

# n = 3^7
# test(n, true)
xs = 2 .^ (2:27)
#xs = collect(1:120)
# xs = sort(vcat([2 .^(2:24), 3 .^(2:15), 5 .^(2:10), 7 .^(2:8), 10 .^(2:7)]...))
benchmark_fft_over_range(xs; ctype=ComplexF32, plan_type=FFTW.MEASURE)
println("Done!")