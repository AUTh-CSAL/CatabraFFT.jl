using CatabraFFT
using BenchmarkTools, FFTW, LinearAlgebra
using Plots

plotlyjs()

# Bencharks CatabraFFT.jl compared to FFTW.jl plots the results.

relative_error(x, y) = norm(x - y) / norm(y)

function fftwplantype2str(plantype)
    if plantype == FFTW.ESTIMATE
        return "FFTW ESTIMATE"
    elseif plantype == FFTW.MEASURE
        return "FFTW MEASURE"
    elseif plantype == FFTW.PATIENT
        return "FFTW PATIENT"
    elseif plantype == FFTW.EXHAUSTIVE
        return "FFTW EXHAUSTIVE"
    else
        return "*** Unknown FFTW Plan Type ***"
    end
end

function bench(n::Int, fftw_time::Vector, mixed_radix_time::Vector, fftw_mem::Vector, mixed_radix_mem::Vector; ctype=ComplexF64, plan_type=FFTW.MEASURE)

    F = FFTW.plan_fft(randn(ctype, n); flags=plan_type, timelimit=Inf)
    x = randn(ctype, n)

    fftw_result = F * x
    y_mixed = similar(x)
    y_mixed = CatabraFFT.fft(x)
    @show rel_err = relative_error(y_mixed, fftw_result)
    @assert y_mixed ≈ fftw_result

    # Benchmark FFTW
    t_fftw = @benchmark $F * x setup = (x = randn($ctype, $n))
    push!(fftw_time, (median(t_fftw).time / 10^9))
    push!(fftw_mem, (median(t_fftw).memory / 1024))

    # Run custom FFT benchmark
    t_mixed = @benchmark $y_mixed = CatabraFFT.fft(x) setup = (x = randn($ctype, $n))
    push!(mixed_radix_time, (median(t_mixed).time / 10^9))
    push!(mixed_radix_mem, (median(t_mixed).memory / 1024))
end

function bench_ivdep(n::Int, ivdep_time::Vector, ivdep_mem::Vector; ctype=ComplexF64, plan_type=FFTW.MEASURE)

    x = randn(ctype, n)

    y_mixed = similar(x)
    y_mixed = CatabraFFT.fft(x, true)

    # Run custom FFT benchmark
    t_mixed = @benchmark $y_mixed = CatabraFFT.fft(x, true) setup = (x = randn($ctype, $n))
    push!(ivdep_time, (median(t_mixed).time / 10^9))
    push!(ivdep_mem, (median(t_mixed).memory / 1024))
end

function benchmark_fft_over_range(xs::Vector; ctype=ComplexF64, plan_type=FFTW.MEASURE, save=false, msg="", use_ivdep::Bool)
    gflops_catabra, gflops_fftw, gflops_ivdep  = [], [], []
    fftw_time, mixed_radix_time, ivdep_time  = [], [], []
    fftw_mem, mixed_radix_mem, ivdep_mem  = [], [], []

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

    if use_ivdep
        CatabraFFT.empty_cache()
        for n in xs
            CatabraFFT.fft(rand(ctype, n), true)
        end

        for n in xs
            print("n = $n \n")
            bench_ivdep(n, ivdep_time, ivdep_mem; ctype, plan_type)
            println(" Time mixed radix: ", ivdep_time[end])
            push!(gflops_ivdep, (5 * n * log2(n) * 10^(-9)) / ivdep_time[end])
        end
    end

    info = Sys.cpu_info()[1]
    cpu = "$(info.model)@$(info.speed) Julia $(VERSION)"
    ptype = fftwplantype2str(plan_type)

    p_reltime = bar(
        log2.(xs), fftw_time ./ mixed_radix_time, label="",
        linestyle=:none, markershape=:square, markercolor=:red, legend=:bottom, fillalpha=0.5)
    if use_ivdep
    bar!(p_reltime,
    log2.(xs), fftw_time ./ ivdep_time, label="",
    linestyle=:none, markershape=:square, markercolor=:purple, legend=:bottom, fillalpha=0.5)
    end

    xlabel!(p_reltime, "log2(n)")
    ylabel!(p_reltime, "Relative Time (FFTW / CatabraFFT)")
    title!(p_reltime, "$ctype $msg Speedup ($cpu)")

    display(p_reltime)
    save && savefig(p_reltime, "svgs/$msg-speedup-$ctype-$ptype-$cpu.svg")

    # p_time = plot(
    #     log2.(xs), log10.(fftw_time), label="$ptype",
    #     linestyle=:solid, markershape=:square, markercolor=:red, legend=:bottomright)
    # plot!(p_time,
    #     log2.(xs), log10.(mixed_radix_time), label="CatabraFFT",
    #     linestyle=:solid, markershape=:circle, markercolor=:orange)

    # xlabel!(p_time, "log2(n)")
    # ylabel!(p_time, "log10(Time (sec))")
    # title!(p_time, "$ctype $msg Time ($cpu)")

    # display(p_time)
    # save && savefig(p_time, "svgs/$msg-time-$ctype-$ptype-$cpu.svg")

    p_gflops = plot(
        log2.(xs), gflops_fftw, label="$(fftwplantype2str(plan_type)) GFLOPS",
        linestyle=:solid, markershape=:square, markercolor=:red, legend=:bottom)
    plot!(p_gflops,
        log2.(xs), gflops_catabra, label="CatabraFFT GFLOPS",
        linestyle=:solid, markershape=:circle, markercolor=:orange)
    if use_ivdep
    plot!(p_gflops,
        log2.(xs), gflops_ivdep, label="CatabraFFT IVDEP GFLOPS",
        linestyle=:solid, markershape=:circle, markercolor=:purple)
    end

    xlabel!(p_gflops, "log2(Input length)")
    ylabel!(p_gflops, "GFLOPS")
    title!(p_gflops, "$ctype $msg GFLOPS ($cpu)")

    display(p_gflops)
    save && savefig(p_gflops, "svgs/$msg-gflops-$ctype-$ptype-$cpu.svg")
end

fftwplan = FFTW.MEASURE
save = false
twoexp = 10
# IVDEP IS VERY SLOW, UNPREDICTABLE AND EXPRERIMENTAL
use_ivdep = true
for b in [2 3 5 7 10]
    xs = b .^ (2:Int64(floor(twoexp / log2(b))))
    for ctype in [ComplexF32, ComplexF64]
        benchmark_fft_over_range(xs; ctype, plan_type=fftwplan, save, msg="FFT-$b", use_ivdep)
    end
end
println("Done!")
