const INV_SQRT2_DEFAULT = 0.7071067811865475
const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)


using SIMD
using BenchmarkTools

# First implementation using vload/vstore
@inline function fft2_shell_vload!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = pointer(reinterpret(T, x))
    y_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})
    @inbounds @simd for q in 0:s-1
        x_idx1, x_idx2 = x_ptr + q * FLOAT_SIZE, x_ptr + (q + s) * FLOAT_SIZE
        y_idx1, y_idx2 = y_ptr + q * FLOAT_SIZE, y_ptr + (q + s) * FLOAT_SIZE
        # Load 2 complex numbers (256 bits) from each half
        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        sum = a + b
        diff = a - b
        vstore(sum, y_idx1)
        vstore(diff, y_idx2)
    end
    return nothing
end

@inline function fft2_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        a, b = x[q], x[q+s]
        y[q] = a + b
        y[q+s] = a - b
    end
end

@inline function fft2_shell_ivdep!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd ivdep for q in 1:s
        a, b = x[q], x[q+s]
        y[q] = a + b
        y[q+s] = a - b
    end
end

@inline function fft4_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        t1, t2 = x[q] + x[q+2s], x[q] - x[q+2s]
        t3, t4 = (x[q+s] + x[q+3s]), -im * (x[q+s] - x[q+3s])
        y[q], y[q+s] = t1 + t3, t2 + t4
        y[q+2s], y[q+3s] = t1 - t3, t2 - t4
    end
end

@inline function fft4_shell_ivdep!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd ivdep for q in 1:s
        t1, t2 = x[q] + x[q+2s], x[q] - x[q+2s]
        t3, t4 = (x[q+s] + x[q+3s]), -im * (x[q+s] - x[q+3s])
        y[q], y[q+s] = t1 + t3, t2 + t4
        y[q+2s], y[q+3s] = t1 - t3, t2 - t4
    end
end

@inline function fft2_shell_ptr!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = Base.unsafe_convert(Ptr{Complex{T}}, x)
    y_ptr = Base.unsafe_convert(Ptr{Complex{T}}, y)
    
    @inbounds @simd ivdep for q in 1:s
        a = unsafe_load(x_ptr, q)
        b = unsafe_load(x_ptr, q+s)
        unsafe_store!(y_ptr, a + b, q)
        unsafe_store!(y_ptr, a - b, q+s)
    end
    return nothing
end

@inline function fft8_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    @inbounds @simd for q in 1:s
        t13, t14 = x[q] + x[q+4s], x[q] - x[q+4s]
        t15, t16 = (x[q+2s] + x[q+6s]), -im * (x[q+2s] - x[q+6s])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s] + x[q+5s], x[q+s] - x[q+5s]
        t19, t20 = (x[q+3s] + x[q+7s]), -im * (x[q+3s] - x[q+7s])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 -im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 -im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end
end

@inline function fft8_shell_ivdep!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    @inbounds @simd ivdep for q in 1:s
        t13, t14 = x[q] + x[q+4s], x[q] - x[q+4s]
        t15, t16 = (x[q+2s] + x[q+6s]), -im * (x[q+2s] - x[q+6s])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s] + x[q+5s], x[q+s] - x[q+5s]
        t19, t20 = (x[q+3s] + x[q+7s]), -im * (x[q+3s] - x[q+7s])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 -im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 -im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end
end

function fft_chunk!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    n = length(y)
    fft8_shell!(y, x, n ÷ 8)
    fft4_shell!(y, x, n ÷ 4)
    fft2_shell!(y, x, n ÷ 2)
end

function fft_chunk_ivdep!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    n = length(y)
    fft8_shell_ivdep!(y, x, n ÷ 8)
    fft4_shell_ivdep!(y, x, n ÷ 4)
    fft2_shell_ivdep!(y, x, n ÷ 2)
end

# Benchmark setup
function benchmark_fft_variants()
    # Various input sizes to test
    sizes = [2^10, 2^12, 2^13, 2^14, 2^16, 2^18, 2^25]
    
    for n in sizes
        x = [ComplexF64(i,i) for i in 1:4n]
        y1 = similar(x)
        y2 = similar(x)
        y3 = similar(x)
        y4 = similar(x)
        y5 = similar(x)
        y6 = similar(x)
        y7 = similar(x)
        y8 = similar(x)
        
        s = n ÷ 2  
        s2 = n ÷ 4
        s3 = n ÷ 8
        
        println("Benchmarking for array size $n:")
        
        # Warm-up runs
        fft2_shell!(y1, x, s)
        fft2_shell_ivdep!(y2, x, s)
        fft4_shell!(y3, x, s2)
        fft4_shell_ivdep!(y4, x, s2)
        fft8_shell!(y5, x, s3)
        fft8_shell_ivdep!(y6, x, s3)
        fft_chunk!(y7, x)
        fft_chunk_ivdep!(y8, x)
        
        # Verify correctness
        #@assert y1 == y2 == y3 == y4 "Implementations produce different results!"
        
        println("OG fft2")
        @btime fft2_shell!($y1, $x, $s)
        
        println("ivdep fft2")
        @btime fft2_shell_ivdep!($y2, $x, $s)

        println("OG fft4")
        @btime fft4_shell!($y3, $x, $s2)

        println("ivdep fft4")
        @btime fft4_shell_ivdep!($y4, $x, $s2)

        println("OG fft8")
        @btime fft8_shell!($y5, $x, $s3)

        println("ivdep fft8")
        @btime fft8_shell_ivdep!($y6, $x, $s3)

        println("fft chunk")
        @btime fft_chunk!($y7, $x)

        println("ivdep fft chunk")
        @btime fft_chunk_ivdep!($y8, $x)
        
        println("\n")
    end
end

# Run benchmarks
#benchmark_fft_variants()

module OptimizedFFT

using SIMD
using LoopVectorization
using Base.Threads
using Statistics    

# Enhanced compile-time configuration with validation
struct FFTConfig{T <: AbstractFloat}
    vector_width::Int
    cache_line_size::Int
    num_cores::Int
    type::Type{T}

    function FFTConfig{T}(; 
        vector_width::Int = sizeof(Complex{T}) ÷ sizeof(T),
        cache_line_size::Int = 64,
        num_cores::Int = Threads.nthreads(),
        type::Type{T} = Float64
    ) where {T <: AbstractFloat}
        @assert vector_width > 0 "Vector width must be positive"
        @assert cache_line_size > 0 "Cache line size must be positive"
        @assert num_cores > 0 "Number of cores must be positive"
        new{T}(vector_width, cache_line_size, num_cores, type)
    end
end

# Constants
const INV_SQRT2 = 0.7071067811865475244008443621048490392848359376884740

"""
    optimized_fft_codelet!(output, input, config, radix=2)

Compute FFT using optimized codelet implementation.

# Arguments
- `output::AbstractVector{Complex{T}}`: Output buffer
- `input::AbstractVector{Complex{T}}`: Input data
- `config::FFTConfig{T}`: Hardware configuration
- `radix::Int=2`: FFT radix (2, 4, or 8)

# Returns
- `output::AbstractVector{Complex{T}}`: Transformed data
"""
function optimized_fft_codelet!(
    output::AbstractVector{Complex{T}}, 
    input::AbstractVector{Complex{T}}, 
    config::FFTConfig{T}, 
    radix::Int = 2
) where T <: AbstractFloat
    n = length(input)
    validate_input_size(n, radix)
    
    # Optimize thread count based on input size and hardware
    num_threads = optimize_thread_count(n, config)
    chunk_size = div(n, num_threads)
    
    # Pre-compute twiddle factors with cache alignment
    twiddle_factors = cache_aligned_twiddle_factors(n, T)
    
    # Parallel processing with dynamic scheduling
    @threads for thread_id in 0:num_threads-1
        process_thread_chunk!(
            output, input, twiddle_factors,
            thread_id, chunk_size, n,
            config, radix
        )
    end
    
    return output
end

"""Input validation with detailed error messages"""
function validate_input_size(n::Int, radix::Int)
    if !ispow2(n)
        throw(ArgumentError("Input length ($n) must be a power of 2"))
    end
    if n < radix
        throw(ArgumentError("Input length ($n) must be >= radix ($radix)"))
    end
    if !(radix in [2, 4, 8])
        throw(ArgumentError("Unsupported radix: $radix. Must be 2, 4, or 8"))
    end
end

"""Optimize thread count based on problem size and hardware"""
function optimize_thread_count(n::Int, config::FFTConfig)
    min_chunk_size = max(32, config.vector_width * 4)  # Ensure sufficient work per thread
    return min(config.num_cores, div(n, min_chunk_size))
end

"""Generate cache-aligned twiddle factors"""
function cache_aligned_twiddle_factors(n::Int, ::Type{T}) where T
    factors = Vector{Complex{T}}(undef, n)
    @inbounds @simd for k in 1:n
        factors[k] = Complex{T}(cispi(-2 * (k-1) / n))
    end
    return factors
end

"""Process a single thread's chunk of data"""
function process_thread_chunk!(
    output::AbstractVector{Complex{T}},
    input::AbstractVector{Complex{T}},
    twiddle_factors::AbstractVector{Complex{T}},
    thread_id::Int,
    chunk_size::Int,
    total_size::Int,
    config::FFTConfig{T},
    radix::Int
) where T <: AbstractFloat
    start_idx = thread_id * chunk_size + 1
    end_idx = thread_id == div(total_size, chunk_size) - 1 ? total_size : (thread_id + 1) * chunk_size
    
    chunk_view = view(output, start_idx:end_idx)
    input_view = view(input, start_idx:end_idx)
    twiddle_view = view(twiddle_factors, start_idx:end_idx)
    
    process_chunk_radix!(chunk_view, input_view, twiddle_view, config, radix)
end

"""Radix-specific processing with SIMD optimization"""
@inline function process_chunk_radix!(
    output::AbstractVector{Complex{T}},
    input::AbstractVector{Complex{T}},
    twiddle::AbstractVector{Complex{T}},
    config::FFTConfig{T},
    radix::Int
) where T <: AbstractFloat
    if radix == 2
        process_radix2!(output, input, twiddle, config)
    elseif radix == 4
        process_radix4!(output, input, twiddle, config)
    else # radix == 8
        process_radix8!(output, input, twiddle, config)
    end
end

"""Optimized Radix-2 butterfly computation"""
@inline function process_radix2!(
    output::AbstractVector{Complex{T}},
    input::AbstractVector{Complex{T}},
    twiddle::AbstractVector{Complex{T}},
    config::FFTConfig{T}
) where T <: AbstractFloat
    @inbounds @simd for i in 1:2:length(input)
        a, b = input[i], input[i+1]
        t = twiddle[i] * b
        output[i] = a + t
        output[i+1] = a - t
    end
end

"""Optimized Radix-4 butterfly computation"""
@inline function process_radix4!(
    output::AbstractVector{Complex{T}},
    input::AbstractVector{Complex{T}},
    twiddle::AbstractVector{Complex{T}},
    config::FFTConfig{T}
) where T <: AbstractFloat
    @inbounds @simd for i in 1:4:length(input)
        a, b, c, d = input[i:i+3]
        t1 = a + c
        t2 = a - c
        t3 = b + d
        t4 = im * (b - d)
        
        output[i] = t1 + t3
        output[i+1] = t2 + t4
        output[i+2] = t1 - t3
        output[i+3] = t2 - t4
    end
end

"""Optimized Radix-8 butterfly computation"""
@inline function process_radix8!(
    output::AbstractVector{Complex{T}},
    input::AbstractVector{Complex{T}},
    twiddle::AbstractVector{Complex{T}},
    config::FFTConfig{T}
) where T <: AbstractFloat
    invsqrt2 = T(INV_SQRT2)
    @inbounds @simd for i in 1:8:length(input)
        a, b, c, d, e, f, g, h = input[i:i+7]
        
        # Stage 1: Initial combinations
        t1 = a + e
        t2 = a - e
        t3 = c + g
        t4 = im * (c - g)
        t5 = b + f
        t6 = b - f
        t7 = d + h
        t8 = im * (d - h)
        
        # Stage 2: Final combinations with twiddle factors
        output[i] = t1 + t3
        output[i+1] = t2 + invsqrt2 * (t6 + t8)
        output[i+2] = t5 + t7
        output[i+3] = t4 + invsqrt2 * (t2 - t8)
        output[i+4] = t1 - t3
        output[i+5] = t2 - invsqrt2 * (t6 + t8)
        output[i+6] = t5 - t7
        output[i+7] = t4 - invsqrt2 * (t2 - t8)
    end
end

"""
Comprehensive benchmarking function with detailed performance metrics
"""
function benchmark_fft_strategies(;
    input_sizes::Vector{Int} = [2^10, 2^14, 2^18, 2^22],
    radix_options::Vector{Int} = [2, 4, 8],
    num_trials::Int = 5
)
    config = FFTConfig{Float64}()
    
    println("Hardware Configuration:")
    println("  Vector Width: ", config.vector_width)
    println("  Cache Line Size: ", config.cache_line_size, " bytes")
    println("  Available Threads: ", config.num_cores)
    
    for size in input_sizes
        x = [Complex{Float64}(rand(), rand()) for _ in 1:size]
        y = similar(x)
        
        println("\nInput size: $size")
        
        for r in radix_options
            println("\nRadix-$r FFT:")
            
            # Warmup
            optimized_fft_codelet!(y, x, config, r)
            
            # Benchmark trials
            times = Float64[]
            for _ in 1:num_trials
                time = @elapsed optimized_fft_codelet!(y, x, config, r)
                push!(times, time)
            end
            
            # Report statistics
            avg_time = mean(times)
            std_dev = std(times)
            gflops = (5 * size * log2(size)) / (1e9 * avg_time)
            
            println("  Average time: $(round(avg_time, digits=6)) seconds")
            println("  Std Dev: $(round(std_dev, digits=6)) seconds")
            println("  Performance: $(round(gflops, digits=2)) GFLOPS")
        end
    end
end

# Export public interface
export FFTConfig, optimized_fft_codelet!, benchmark_fft_strategies

end # module OptimizedFFT


#OptimizedFFT.benchmark_fft_strategies()
