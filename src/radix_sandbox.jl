const INV_SQRT2_DEFAULT = 0.7071067811865475
const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)

#=
module FFTOptimizer

using SIMD, LoopVectorization

# Compile-time configuration
struct SIMDConfig{T}
    vector_width::Int
    element_size::Int
    # Add more machine-specific parameters
end

# Construct configuration at compile time
function configure_simd(::Type{T}) where T
    return SIMDConfig{T}(
        get_vector_width(T),
        sizeof(T)
    )
end

# Use the configuration in your FFT implementation
@inline function fft2_shell_adaptive!(
    y::AbstractVector{Complex{T}}, 
    x::AbstractVector{Complex{T}}, 
    s::Int,
    config::SIMDConfig{T}
) where T <: AbstractFloat
    # Use config.vector_width, config.element_size etc.
    # Implement your vectorized logic here
end

# Determine vector width based on SIMD.jl capabilities
function get_vector_width(::Type{T}) where T
    # Use SIMD.jl's Vec type to determine natural vector width
    return length(Vec{:auto, T})
end

# Determine element size
function get_element_size(::Type{T}) where T
    return sizeof(T)
end

@inline function fft2_shell_adaptive_64!(
    y::AbstractVector{Complex{Float64}}, 
    x::AbstractVector{Complex{Float64}}, 
    s::Int
)
    # Reinterpret complex vectors to Float64
    x_float = reinterpret(Float64, x)
    y_float = reinterpret(Float64, y)
    
    # Detect vector length dynamically
    VEC_LEN = get_vector_width(Float64)
    FLOAT_SIZE = get_element_size(Float64)
    
    # Pointer conversion
    x_ptr = pointer(x_float)
    y_ptr = pointer(y_float)
    
    # Adaptive implementation based on stride size
    if s == 1
        # Continuous memory access case - direct vload/vstore
        @inbounds @simd for q in 0:div(length(x), VEC_LEN)-1
            # Load VEC_LEN elements at once
            a = vload(Vec{VEC_LEN, Float64}, x_ptr + q * VEC_LEN * FLOAT_SIZE)
            # Perform transformation (could be generalized)
            transformed = a  # Replace with actual transformation logic
            vstore(transformed, y_ptr + q * VEC_LEN * FLOAT_SIZE)
        end
    else
        # Non-continuous memory access - use vgather/vscatter
        @inbounds for q in 0:s-1
            # Compute gather/scatter indices
            gather_indices = Vec((q, q + s))
            #gather_indices = Vec(ntuple(i -> q + (i-1)*s, Val(VEC_LEN)))
            #gather_indices = Vec{VEC_LEN, Int}(ntuple(i -> convert(Int, q + (i-1)*s), Val(VEC_LEN)))
            
            # Perform vectorized gather
            a = vgather(x_float, gather_indices)
            
            # Compute FFT2 shell transformation
            transformed = a  # Replace with actual transformation logic
            
            # Vectorized scatter
            vscatter(transformed, y_float, gather_indices)
        end
    end
    
    return nothing
end

# Specialized Cooley-Tukey FFT2 shell transform
@inline function fft2_shell_cooley_tukey_64!(
    y::AbstractVector{Complex{Float64}}, 
    x::AbstractVector{Complex{Float64}}, 
    s::Int
)
    x_float = reinterpret(Float64, x)
    y_float = reinterpret(Float64, y)
    
    @inbounds for q in 0:s-1
        # Compute complex number indices
        x_idx1 = 2(q) + 1
        x_idx2 = 2(q + s) + 1
        
        # Load real and imaginary parts
        a_real = x_float[x_idx1]
        a_imag = x_float[x_idx1 + 1]
        b_real = x_float[x_idx2]
        b_imag = x_float[x_idx2 + 1]
        
        # Cooley-Tukey butterfly computation
        sum_real = a_real + b_real
        sum_imag = a_imag + b_imag
        diff_real = a_real - b_real
        diff_imag = a_imag - b_imag
        
        # Store transformed values
        y_float[x_idx1] = sum_real
        y_float[x_idx1 + 1] = sum_imag
        y_float[x_idx2] = diff_real
        y_float[x_idx2 + 1] = diff_imag
    end
    
    return nothing
end

# Benchmark and performance testing function
function benchmark_fft2_shell(n::Int, s::Int)
    x = randn(Complex{Float64}, n)
    y = similar(x)
    
    # Warmup
    fft2_shell_adaptive_64!(y, x, s)
    
    # Timing
    @time fft2_shell_adaptive_64!(y, x, s)
    @time fft2_shell_cooley_tukey_64!(y, x, s)
    
    return y
end


@inline function fft2_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
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

@inline function fft2_shell_vgather!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    # Convert to float representation
    x_float = reinterpret(T, x)
    y_float = reinterpret(T, y)
    
    @inbounds @simd for q in 1:s
        # Create gather indices for the two complex numbers
        # Each complex number is 2 consecutive float elements
        idx1 = Vec((2(q-1) + 1, 2(q+s-1) + 1))
        idx2 = Vec((2(q-1) + 2, 2(q+s-1) + 2))
        
        # Vectorized gather operation
        real = vgather(x_float, idx1)
        imag = vgather(x_float, idx2)
        
        # Compute butterfly transformation
        sum = Vec(real[1] + real[2], imag[1] + imag[2])  # Real + Real, Imag + Imag
        diff = Vec(real[1] - real[2], imag[1] - imag[2])  # Real - Real, Imag - Imag
        
        # Scatter transformed values
        vscatter(sum, y_float, idx1)
        vscatter(diff, y_float, idx2)
    end
    
    return nothing
end


# Example usage
x = [ComplexF64(i,i) for i in 1:16]
y = similar(x)
fft2_shell_vgather!(y, x, 1)
@show y


end
=#



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
benchmark_fft_variants()

module OptimizedFFT

using SIMD
using LoopVectorization
using Base.Threads

# Compile-time configuration for hardware-specific optimization
struct FFTConfig{T}
    vector_width::Int
    cache_line_size::Int
    num_cores::Int
    precision::Type{T}
end

# Constant for inverse sqrt(2)
const INV_SQRT2 = 0.7071067811865475

"""
Adaptive FFT Codelet with multiple optimization strategies
- Cache-friendly memory access
- SIMD vectorization
- Multi-threading support
- Adaptive to different input sizes and hardware
"""
@inline function optimized_fft_codelet!(
    output::AbstractVector{Complex{T}}, 
    input::AbstractVector{Complex{T}}, 
    config::FFTConfig{T}, 
    radix::Int = 2
) where T <: AbstractFloat
    # Input validation
    n = length(input)
    @assert ispow2(n) "Input length must be a power of 2"
    
    # Determine optimal threading strategy
    num_threads = min(config.num_cores, div(n, config.vector_width))
    chunk_size = div(n, num_threads)
    
    # Twiddle factor pre-computation (cache-friendly)
    twiddle_factors = precompute_twiddle_factors(n, T)
    
    # Multi-threaded FFT computation
    Threads.@threads for thread_id in 0:num_threads-1
        thread_start = thread_id * chunk_size + 1
        thread_end = thread_id == num_threads-1 ? n : (thread_id + 1) * chunk_size
        
        # Local cache-friendly processing
        process_chunk!(
            view(output, thread_start:thread_end), 
            view(input, thread_start:thread_end), 
            view(twiddle_factors, thread_start:thread_end), 
            config, 
            radix
        )
    end
    
    return output
end

"""
Precompute twiddle factors with cache considerations
"""
function precompute_twiddle_factors(n::Int, ::Type{T}) where T
    factors = Vector{Complex{T}}(undef, n)
    @inbounds @simd for k in 1:n
        angle = -2π * (k-1) / n
        factors[k] = Complex{T}(cos(angle), sin(angle))
    end
    return factors
end

"""
Process a chunk of the FFT with SIMD and cache-friendly strategies
"""
@inline function process_chunk!(
    output_chunk::AbstractVector{Complex{T}}, 
    input_chunk::AbstractVector{Complex{T}}, 
    twiddle_chunk::AbstractVector{Complex{T}}, 
    config::FFTConfig{T}, 
    radix::Int
) where T <: AbstractFloat
    # Adaptive SIMD processing based on hardware configuration
    vector_width = config.vector_width
    
    # SIMD butterfly computation with different radix strategies
    if radix == 2
        @inbounds @simd ivdep for i in 1:2:length(input_chunk)
            # Butterfly computation with SIMD-friendly access
            a = input_chunk[i]
            b = input_chunk[i+1]
            
            # Complex number butterfly operation
            output_chunk[i] = a + b
            output_chunk[i+1] = a - b
        end
    elseif radix == 4
        @inbounds @simd ivdep for i in 1:4:length(input_chunk)
            # 4-way butterfly with more complex computation
            t1 = input_chunk[i] + input_chunk[i+2]
            t2 = input_chunk[i] - input_chunk[i+2]
            t3 = input_chunk[i+1] + input_chunk[i+3]
            t4 = im * (input_chunk[i+1] - input_chunk[i+3])
            
            output_chunk[i]   = t1 + t3
            output_chunk[i+1] = t2 + t4
            output_chunk[i+2] = t1 - t3
            output_chunk[i+3] = t2 - t4
        end
    elseif radix == 8
        INV_SQRT2 = T(0.7071067811865475)
        @inbounds @simd ivdep for i in 1:8:length(input_chunk)
            # Complex 8-way butterfly with advanced transformations
            t1 = input_chunk[i] + input_chunk[i+4]
            t2 = input_chunk[i] - input_chunk[i+4]
            t3 = input_chunk[i+2] + input_chunk[i+6]
            t4 = im * (input_chunk[i+2] - input_chunk[i+6])
            
            t5 = input_chunk[i+1] + input_chunk[i+5]
            t6 = input_chunk[i+1] - input_chunk[i+5]
            t7 = input_chunk[i+3] + input_chunk[i+7]
            t8 = im * (input_chunk[i+3] - input_chunk[i+7])
            
            # Advanced butterfly computations with twiddle factor considerations
            output_chunk[i]   = t1 + t3
            output_chunk[i+1] = t2 + INV_SQRT2 * (t6 + t8)
            output_chunk[i+2] = t5 + t7
            output_chunk[i+3] = t4 + INV_SQRT2 * (t2 - t8)
            
            # Remaining outputs with complex transformations
            output_chunk[i+4] = t1 - t3
            output_chunk[i+5] = t2 - INV_SQRT2 * (t6 + t8)
            output_chunk[i+6] = t5 - t7
            output_chunk[i+7] = t4 - INV_SQRT2 * (t2 - t8)
        end
    else
        throw(ArgumentError("Unsupported radix: $radix"))
    end
end

"""
Benchmark function to compare different FFT implementation strategies
"""
function benchmark_fft_strategies(
    input_sizes::Vector{Int} = [2^10, 2^14, 2^18, 2^22],
    radix_options::Vector{Int} = [2, 4, 8]
)
    # Detect hardware configuration
    config = FFTConfig{Float64}(
        vector_width = length(Vec{:auto, Float64}),
        cache_line_size = 64,  # Typical cache line size
        num_cores = Threads.nthreads(),
        precision = Float64
    )
    
    println("Hardware Configuration:")
    println("  Vector Width: ", config.vector_width)
    println("  Cache Line Size: ", config.cache_line_size, " bytes")
    println("  Available Threads: ", config.num_cores)
    
    for size in input_sizes
        x = [ComplexF64(rand(), rand()) for _ in 1:size]
        y = similar(x)
        
        println("\nBenchmarking for input size: ", size)
        
        for r in radix_options
            println("\nRadix-", r, " FFT:")
            
            # Warm-up run
            optimized_fft_codelet!(y, x, config, r)
            
            # Actual timing
            @time optimized_fft_codelet!(y, x, config, r)
        end
    end
end

# Example usage
function main()
    benchmark_fft_strategies()
end

# Export the main functions
export optimized_fft_codelet!, benchmark_fft_strategies

end  # module OptimizedFFT

module RadixFFTCompositor

using LoopVectorization, SIMD

const INV_SQRT2_DEFAULT = 0.7071067811865475
const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)

# Define kernel patterns and suffixes
const KERNEL_PATTERNS = Dict(
    2 => """
        a, b = #INPUT#[q], #INPUT#[q+s]
        #OUTPUT#[q] = a + b
        #OUTPUT#[q+s] = a - b
    """,
    4 => """
        t1, t2 = #INPUT#[q] + #INPUT#[q+2s], #INPUT#[q] - #INPUT#[q+2s]
        t3, t4 = (#INPUT#[q+s] + #INPUT#[q+3s]), -im * (#INPUT#[q+s] - #INPUT#[q+3s])
        #OUTPUT#[q], #OUTPUT#[q+s] = t1 + t3, t2 + t4
        #OUTPUT#[q+2s], #OUTPUT#[q+3s] = t1 - t3, t2 - t4
    """,
    8 => """
        t13, t14 = #INPUT#[q] + #INPUT#[q+4s], #INPUT#[q] - #INPUT#[q+4s]
        t15, t16 = (#INPUT#[q+2s] + #INPUT#[q+6s]), -im * (#INPUT#[q+2s] - #INPUT#[q+6s])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = #INPUT#[q+s] + #INPUT#[q+5s], #INPUT#[q+s] - #INPUT#[q+5s]
        t19, t20 = (#INPUT#[q+3s] + #INPUT#[q+7s]), -im * (#INPUT#[q+3s] - #INPUT#[q+7s])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 - im) * (t18 - t20)

        #OUTPUT#[q], #OUTPUT#[q+s], #OUTPUT#[q+2s], #OUTPUT#[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        #OUTPUT#[q+4s], #OUTPUT#[q+5s], #OUTPUT#[q+6s], #OUTPUT#[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    """
)

# Function to generate kernel name
function generate_kernel_name(radix::Int, suffixes::Vector{String})
    base = "fft$(radix)_shell"
    suffix = join(suffixes, "_")
    return string(base, !isempty(suffix) ? "_$suffix" : "", "!")
end

# Function to generate function signature
function generate_signature(radix::Int, suffixes::Vector{String})
    y_only = "y" in suffixes
    layered = "layered" in suffixes
    
    if y_only
        return "(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat"
    elseif layered
        return "(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int, n1::Int, theta::Float64=0.125) where T <: AbstractFloat"
    else
        return "(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat"
    end
end

# Function to generate loop decorators
function generate_loop_decorators(suffixes::Vector{String})
    decorators = ["@inbounds"]
    if "ivdep" in suffixes
        push!(decorators, "@simd ivdep")
    else
        push!(decorators, "@simd")
    end
    return join(decorators, " ")
end

# Main function to generate kernel code
function generate_kernel(radix::Int, suffixes::Vector{String})
    name = generate_kernel_name(radix, suffixes)
    signature = generate_signature(radix, suffixes)
    decorators = generate_loop_decorators(suffixes)
    
    kernel_pattern = KERNEL_PATTERNS[radix]
    input = "y" in suffixes ? "y" : "x"
    output = "y"
    
    # Replace placeholders in kernel pattern
    kernel_code = replace(kernel_pattern, 
        "#INPUT#" => input,
        "#OUTPUT#" => output)
    
    # Generate the complete function
    if "layered" in suffixes
        # Special handling for layered kernels
        return generate_layered_kernel(name, signature, decorators, kernel_code, radix)
    else
        return """
        @inline function $name$signature
            INV_SQRT2 = T(INV_SQRT2_DEFAULT)
            $decorators for q in 1:s
                $kernel_code
            end
        end
        """
    end
end

# Helper function to generate layered kernel
function generate_layered_kernel(name, signature, decorators, kernel_code, radix)
    # Additional constants for radix-8
    constants = radix == 8 ? """
        INV_SQRT2 = T(INV_SQRT2_DEFAULT)
        Cp_3_8 = T(Cp_3_8_DEFAULT)
        Sp_3_8 = T(Sp_3_8_DEFAULT)
    """ : "INV_SQRT2 = T(INV_SQRT2_DEFAULT)"
    
    return """
    @inline function $name$signature
        $constants
        
        # First section without twiddle factors
        $decorators for q in 1:s
            $kernel_code
        end
        
        # Section with twiddle factors
        $decorators for p in 1:(n1-1)
            w1p = cispi(T(-p * theta))
            w2p = w1p * w1p
            w3p = w1p * w2p
            $(radix ≥ 4 ? "w4p = w2p * w2p" : "")
            $(radix == 8 ? """
                w5p = w2p * w3p
                w6p = w3p * w3p
                w7p = w3p * w4p
            """ : "")
            
            $decorators for q in 1:s
                $kernel_code
            end
        end
    end
    """
end

# Function to generate all possible kernel combinations
function generate_all_kernels(radixes=[2, 4, 8])
    suffix_combinations = [
        String[],
        ["ivdep"],
        ["y"],
        ["y", "ivdep"],
        ["layered"],
        ["layered", "ivdep"]
    ]
    
    kernels = Dict{String, String}()
    
    for radix in radixes
        for suffixes in suffix_combinations
            name = generate_kernel_name(radix, suffixes)
            code = generate_kernel(radix, suffixes)
            kernels[name] = code
        end
    end
    
    return kernels
end

# Function to evaluate and create the functions in a module
function create_kernel_module()
    kernels = generate_all_kernels()
    
    module_code = """
    module RadixKernels
        using LoopVectorization, SIMD
        
        const INV_SQRT2_DEFAULT = 0.7071067811865475
        const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
        const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)
        
        $(join(values(kernels), "\n\n"))
    end
    """
    
    return module_code
end

end