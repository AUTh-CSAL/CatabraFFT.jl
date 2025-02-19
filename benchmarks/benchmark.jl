using CatabraFFT
using BenchmarkTools, FFTW, LinearAlgebra
using Plots

plotlyjs()

# Bencharks CatabraFFT.jl compared to FFTW.jl plots the results.

relative_error(x, y) = norm(x - y) / norm(y)

function catabraplantype2str(plantype)
    if plantype == CatabraFFT.NO_FLAG
        return "CatabraFFT NO_FLAG" 
    elseif plantype == CatabraFFT.MEASURE
        return "CatabraFFT MEASURE"
    elseif plantype == CatabraFFT.ENCHANT
        return "CatabraFFT ENCHANT"
    else
        return "*** Unknown CatabraFFT Plan Type ***"
    end
end

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

function bench_fftw(n::Int, fftw_time::Vector, fftw_mem::Vector, ctype=ComplexF64, fftw_plan_type=FFTW.MEASURE)

    x = randn(ctype, n)
    F = FFTW.plan_fft(x; flags=fftw_plan_type, timelimit=Inf)

    # Benchmark FFTW
    t_fftw = @benchmark $F * x setup = (x = randn($ctype, $n))
    push!(fftw_time, (median(t_fftw).time / 10^9))
    push!(fftw_mem, (median(t_fftw).memory / 1024))
    
end

function bench_catabra(n::Int, catabra_time::Vector, catabra_mem::Vector, ctype=ComplexF64, fftw_plan_type=FFTW.MEASURE, catabra_plan_type=CatabraFFT.NO_FLAG)

    x = randn(ctype, n)

    C = CatabraFFT.plan_fft(x, catabra_plan_type)
    F = FFTW.plan_fft(x; flags=fftw_plan_type, timelimit=Inf)

    fftw_result = F * x
    catabra_result = C * x
    #@show catabra_result
    rel_err = relative_error(catabra_result, fftw_result)
    @assert catabra_result ≈ fftw_result

    # Run custom FFT benchmark
    t_catabra = @benchmark $C * x setup = (x = randn($ctype, $n))
    push!(catabra_time, (median(t_catabra).time / 10^9))
    push!(catabra_mem, (median(t_catabra).memory / 1024))
end

function benchmark_fft_over_range(xs::Vector; ctype=ComplexF64, fftw_plan_type=FFTW.MEASURE, save=false, msg="")
    # Initialize arrays for each plan type
    #catabraplans = [CatabraFFT.NO_FLAG, CatabraFFT.MEASURE, CatabraFFT.ENCHANT]
    catabraplans = [CatabraFFT.ENCHANT]

    n_plans = length(catabraplans)  # Number of different Catabra plans
    gflops_catabra = [Float64[] for _ in 1:n_plans]
    gflops_fftw = []
    fftw_time = []
    catabra_time = [Float64[] for _ in 1:n_plans]
    fftw_mem = []
    catabra_mem = [Float64[] for _ in 1:n_plans]
    
    
    for n in xs
    bench_fftw(n, fftw_time, fftw_mem, ctype, fftw_plan_type)
    push!(gflops_fftw, (5 * n * log2(n) * 10^(-9)) / fftw_time[end])
    println(" n = $n FFTW Time: ", fftw_time[end])
    end

    for (i, cat_plan) in enumerate(catabraplans)
        # Precompute all functions before benchmarking
        for n in xs
            CatabraFFT.plan_fft(rand(ctype, n), cat_plan)
        end
        
        for n in xs
            print("n = $n \n")
            bench_catabra(n, catabra_time[i], catabra_mem[i], ctype, fftw_plan_type, cat_plan)
            println(" n = $n ", catabraplantype2str(cat_plan), " Time: ", catabra_time[i][end])
            
            # Calculate GFLOPS
            push!(gflops_catabra[i], (5 * n * log2(n) * 10^(-9)) / catabra_time[i][end])
        end
    end
    
    # System info
    info = Sys.cpu_info()[1]
    cpu = "$(info.model)@$(info.speed) Julia $(VERSION)"
    ptype = fftwplantype2str(fftw_plan_type)
    
    # Relative time plot
    p_reltime = bar(
        log2.(xs), 
        [fftw_time ./ catabra_time[i] for i in 1:n_plans],  # Matrix form for grouped bar plot
        #label=["NO FLAG" "MEASURE" "ENCHANT"],  # Set labels for each bar group
        label=["ENCHANT"],  # Set labels for each bar group
        linestyle=:none,
        markershape=:square,
        markercolor=[:red :blue :green],  # Assign distinct colors
        fillalpha=0.5,  # Transparency for overlap
        bar_width=0.7,  # Adjust width if needed for clarity
        legend=:bottom
    )
    xlabel!(p_reltime, "log2(n)")
    ylabel!(p_reltime, "Relative Time (FFTW / CatabraFFT)")
    title!(p_reltime, "$ctype $msg Speedup ($cpu)")
    display(p_reltime)
    save && savefig(p_reltime, "svgs/$msg-speedup-$ctype-$ptype-$cpu.svg")
    
    # GFLOPS plot
    p_gflops = plot(
        log2.(xs),
        gflops_fftw,
        label="$(fftwplantype2str(fftw_plan_type)) GFLOPS",
        linestyle=:solid,
        markershape=:square,
        markercolor=:red,
        legend=:bottom
    )
    
    plot_labels = ["NO FLAG", "MEASURE", "ENCHANT"]
    plot_colors = [:orange, :blue, :green]
    
    for i in 1:n_plans
        plot!(p_gflops,
            log2.(xs),
            gflops_catabra[i],
            label="CatabraFFT $(plot_labels[i]) GFLOPS",
            linestyle=:solid,
            markershape=:circle,
            markercolor=plot_colors[i]
        )
    end
    
    xlabel!(p_gflops, "log2(Input length)")
    ylabel!(p_gflops, "GFLOPS")
    title!(p_gflops, "$ctype $msg GFLOPS ($cpu)")
    display(p_gflops)
    save && savefig(p_gflops, "svgs/$msg-gflops-$ctype-$ptype-$cpu.svg")
end


fftwplan = FFTW.PATIENT
save = false
twoexp = 9
#for b in [2 3 5 7 10]
for b in [2]
    xs = b .^ (1:Int64(floor(twoexp / log2(b))))
    #for ctype in [ComplexF32, ComplexF64]
    for ctype in [ComplexF64]
        benchmark_fft_over_range(xs; ctype, fftw_plan_type=fftwplan, save, msg="FFT-$b")
    end
end

println("Done!")
