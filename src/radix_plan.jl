module Radix_Plan

using LoopVectorization, AbstractFFTs

include("radix_2_codelets.jl")
include("radix_3_codelets.jl")
include("radix_5_codelets.jl")
include("radix_7_codelets.jl")

export RadixPlan

# Struct to hold a single FFT operation
struct FFTOp
    op_type::Symbol  # :fft16, :fft8, :fft4, :fft3,  :fft2
    input_buffer::Symbol  # :x or :y
    output_buffer::Symbol # :x or :y
    stride::Int
    n_groups::Int
    eo::Bool
    #ivdep::Bool
end

# Struct to hold the complete FFT plan
struct RadixPlan{T<:AbstractFloat} <: AbstractFFTs.Plan{T}
    operations::Vector{FFTOp}
    n::Int
end

function benchmark_ivdep_performance(fft_standard!, fft_ivdep!, n, s)
    ivdep_speedup_ratios = []
    
    for n in sizes
        x = [ComplexF64(i,i) for i in 1:n]
        y1 = similar(x)
        y2 = similar(x)
        
        s = n ÷ 2  # Half the array size for butterfly operation
        
        # Warm-up runs
        fft_standard!(y1, x, s)
        fft_ivdep!(y2, x, s)
        
        # Verify correctness
        @assert y1 == y2 "Implementations produce different results!"
        
        # Benchmark standard version
        standard_time = @elapsed begin
            for _ in 1:10
                fft_standard!(y1, x, s)
            end
        end
        
        # Benchmark ivdep version
        ivdep_time = @elapsed begin
            for _ in 1:10
                fft_ivdep!(y2, x, s)
            end
        end
        
        # Calculate speedup ratio
        speedup_ratio = standard_time / ivdep_time
        push!(ivdep_speedup_ratios, speedup_ratio)
        
        println("Array size $n:")
        println("Standard time: $standard_time")
        println("IVDEP time: $ivdep_time")
        println("Speedup ratio: $speedup_ratio")
    end
    
    return ivdep_speedup_ratios
end

function determine_ivdep_threshold(fft_standard!, fft_ivdep!, remaining_n, stride)
    speedup_ratios = benchmark_ivdep_performance(fft_standard!, fft_ivdep!, remaining_n, stride)
    
    # Calculate mean speedup and standard deviation
    mean_speedup = mean(speedup_ratios)
    std_speedup = std(speedup_ratios)
    
    # Determine if IVDEP is consistently beneficial
    is_ivdep_beneficial = mean_speedup > 1.05  # 5% speedup threshold
    
    println("Mean Speedup: $mean_speedup")
    println("Speedup Std Dev: $std_speedup")
    println("IVDEP Beneficial: $is_ivdep_beneficial")
    
    return is_ivdep_beneficial
end

function create_radix_2_plan(n::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1
        # Choose largest possible radix (8, 4, or 2)
        #if remaining_n % 16 == 0 && remaining_n ≥ 16
        #    radix = 16
        if remaining_n % 8 == 0 && remaining_n ≥ 8
            radix = 8
        elseif remaining_n % 4 == 0 && remaining_n ≥ 4
            radix = 4
        else
            radix = 2
        end

        # Add FFT layer
        push!(operations, FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
            #ivdep
        ))

        # Update for next iteration
        remaining_n = remaining_n ÷ radix
        current_stride *= radix
        eo = !eo # swap
        input_buffer, output_buffer = output_buffer, input_buffer
        #ivdep = determine_ivdep_threshold()
    end

    RadixPlan{T}(operations, n)
end

function create_radix_3_plan(n::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1
        if remaining_n % 9 == 0 && remaining_n ≥ 9
            radix = 9
        else
            radix = 3
        end

        # Add FFT layer
        push!(operations, FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
            #ivdep
        ))

        # Update for next iteration
        remaining_n = remaining_n ÷ radix
        current_stride *= radix
        eo = !eo # swap
        input_buffer, output_buffer = output_buffer, input_buffer
    end

    RadixPlan{T}(operations, n)
end

# Simple universal planner for basic modulo radix planning
function create_radix_plan(n::Int, radix::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false
    #ivdep = false

    while remaining_n > 1

        # Add FFT layer
        push!(operations, FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
            #ivdep
        ))

        # Update for next iteration
        remaining_n = remaining_n ÷ radix
        current_stride *= radix
        eo = !eo # swap
        input_buffer, output_buffer = output_buffer, input_buffer
        #ivdep = determin
    end

    RadixPlan{T}(operations, n)
end

end
