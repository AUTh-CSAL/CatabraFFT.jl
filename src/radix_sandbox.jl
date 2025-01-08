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

#OptimizedFFT.benchmark_fft_strategies()
module CacheAwareFFT

using StaticArrays
using Hwloc

# Enhanced cache information structure with more detailed parameters
struct CacheInfo
    l1d_size::Vector{Int}
    l1d_linesize::Int
    l1d_associativity::Int
    l2_size::Int
    l2_linesize::Int
    l2_associativity::Int
    l3_size::Int
    l3_linesize::Int
    vector_size::Int  # CPU vector register size
end

"""
    detect_cache_info() -> CacheInfo

Detect CPU cache parameters using Hwloc with improved error handling and fallbacks.
Returns a CacheInfo struct containing cache hierarchy details.
"""
function detect_cache_info()
    # Initialize topology
    topology = Hwloc.topology_load()
    
    # Get cache sizes using direct Hwloc functions
    l1_sizes = Hwloc.l1cache_sizes(topology)
    l2_sizes = Hwloc.l2cache_sizes(topology)
    l3_sizes = Hwloc.l3cache_sizes(topology)
    
    # Get cache line sizes
    l1_lines = Hwloc.l1cache_linesizes(topology)
    l2_lines = Hwloc.l2cache_linesizes(topology)
    l3_lines = Hwloc.l3cache_linesizes(topology)
    
    # Helper function to safely get first value from vector or use fallback
    function safe_first(vec, fallback)
        isempty(vec) ? fallback : first(vec)
    end
    
    # Extract values with fallbacks
    l1d_size = safe_first(l1_sizes, 32768)  # 32KB fallback
    l1d_linesize = safe_first(l1_lines, 64)  # 64B fallback
    l1d_assoc = 8  # Common fallback for L1 associativity
    
    l2_size = safe_first(l2_sizes, 262144)  # 256KB fallback
    l2_linesize = safe_first(l2_lines, 64)
    l2_assoc = 8  # Common fallback for L2 associativity
    
    l3_size = safe_first(l3_sizes, 8388608)  # 8MB fallback
    l3_linesize = safe_first(l3_lines, 64)
    
    # Detect vector size based on CPU features
    # TODO: Add actual CPU feature detection
    vector_size = 32  # 256 bits = 32 bytes (AVX2)
    
    CacheInfo(
        l1d_size, l1d_linesize, l1d_assoc,
        l2_size, l2_linesize, l2_assoc,
        l3_size, l3_linesize, vector_size
    )
end

"""
    calculate_chunk_sizes(cache_info::CacheInfo, T::Type) -> NamedTuple

Calculate optimal chunk sizes for FFT processing based on cache parameters.
Takes into account:
- Cache line size and associativity
- Vector register width
- Complex number size
- Power of 2 constraints for FFT

Returns a NamedTuple containing:
- chunk_size: Optimal size for processing chunks
- l1_elements: Number of elements that fit in L1 cache
- line_elements: Elements per cache line
- vector_elements: Elements per vector register
"""
function calculate_chunk_sizes(cache_info::CacheInfo, T::Type)
    element_size = sizeof(Complex{T})
    
    # Calculate various capacity constraints
    elements_per_line = cache_info.l1d_linesize ÷ element_size
    vector_elements = cache_info.vector_size ÷ element_size
    
    # Calculate L1 cache capacity accounting for associativity
    # Use 75% of L1 capacity to avoid cache thrashing
    effective_l1_size = (cache_info.l1d_size * 3) ÷ 4
    l1_capacity = effective_l1_size ÷ element_size
    
    # Ensure chunk size is:
    # 1. Multiple of vector size for efficient SIMD
    # 2. Aligned with cache lines
    # 3. Power of 2 for FFT
    # 4. Small enough to fit in L1 cache with room for twiddle factors
    base_chunk = lcm(vector_elements, elements_per_line)
    max_chunk = min(
        l1_capacity ÷ 4,  # Leave room for input/output/twiddle factors
        nextpow(2, base_chunk)
    )
    
    # Find largest power of 2 that satisfies all constraints
    chunk_size = prevpow(2, max_chunk)
    
    return (
        chunk_size = chunk_size,
        l1_elements = l1_capacity,
        line_elements = elements_per_line,
        vector_elements = vector_elements,
        l2_elements = (cache_info.l2_size * 3) ÷ (4 * element_size)
    )
end

# Add utility functions for optimal memory access patterns
"""
    get_optimal_stride(chunk_size::Int, cache_info::CacheInfo) -> Int

Calculate optimal stride for accessing memory to minimize cache conflicts.
"""
function get_optimal_stride(chunk_size::Int, cache_info::CacheInfo)
    # Choose stride to avoid cache line conflicts
    # Should be coprime with cache associativity
    base_stride = chunk_size ÷ cache_info.l1d_associativity
    return nextprime(base_stride)
end

end # module


info = CacheAwareFFT.detect_cache_info()
CacheAwareFFT.calculate_chunk_sizes(info, Float32)

module CacheAwareFFTKernel

using SIMD
using StaticArrays
using Base.Threads

# Re-export cache detection
using ..CacheAwareFFT: CacheInfo, detect_cache_info, calculate_chunk_sizes

const INV_SQRT2 = Float64(1/sqrt(2))
const Cp_3_8 = Float64(cos(π*3/8))  # cospi(3/8)
const Sp_3_8 = Float64(sin(π*3/8))  # sinpi(3/8)

"""
Optimized radix-4 FFT kernel leveraging cache parameters and SIMD
"""
@inline function radix4_kernel_optimized!(
    y::AbstractVector{Complex{T}}, 
    x::AbstractVector{Complex{T}},
    s::Int,
    cache_params::NamedTuple,
    stride::Int = 1
) where T <: AbstractFloat
    chunk_size = cache_params.chunk_size
    vector_elements = cache_params.vector_elements
    
    # Process in cache-friendly chunks
    @inbounds for base in 1:chunk_size:s
        # Calculate end of current chunk
        chunk_end = min(base + chunk_size - 1, s)
        chunk_len = chunk_end - base + 1
        
        # Pre-fetch data into L1 cache using aligned loads
        chunk_0 = Vector{Complex{T}}(undef, chunk_len)
        chunk_1 = Vector{Complex{T}}(undef, chunk_len)
        chunk_2 = Vector{Complex{T}}(undef, chunk_len)
        chunk_3 = Vector{Complex{T}}(undef, chunk_len)
        
        # Vectorized load with stride handling
        @inbounds @simd for i in 1:chunk_len
            idx = base + i - 1
            chunk_0[i] = x[idx]
            chunk_1[i] = x[idx + s]
            chunk_2[i] = x[idx + 2s]
            chunk_3[i] = x[idx + 3s]
        end
        
        # Process chunk with SIMD
        @inbounds @simd for i in 1:chunk_len
            # Radix-4 butterfly operations
            t1 = chunk_0[i] + chunk_2[i]
            t2 = chunk_0[i] - chunk_2[i]
            t3 = chunk_1[i] + chunk_3[i]
            t4 = -im * (chunk_1[i] - chunk_3[i])
            
            idx = base + i - 1
            y[idx] = t1 + t3
            y[idx + s] = t2 + t4
            y[idx + 2s] = t1 - t3
            y[idx + 3s] = t2 - t4
        end
    end
end

"""
Optimized radix-8 FFT kernel with cache and SIMD optimizations
"""
@inline function radix8_kernel_optimized!(
    y::AbstractVector{Complex{T}}, 
    x::AbstractVector{Complex{T}},
    s::Int,
    cache_params::NamedTuple,
    stride::Int = 1
) where T <: AbstractFloat
    chunk_size = cache_params.chunk_size
    vector_elements = cache_params.vector_elements
    
    # Process in cache-friendly chunks
    @inbounds for base in 1:chunk_size:s
        chunk_end = min(base + chunk_size - 1, s)
        chunk_len = chunk_end - base + 1
        
        # Pre-fetch data into L1 cache
        chunks = [Vector{Complex{T}}(undef, chunk_len) for _ in 1:8]
        
        # Vectorized load with stride handling
        @inbounds @simd for i in 1:chunk_len
            idx = base + i - 1
            for j in 0:7
                chunks[j+1][i] = x[idx + j*s]
            end
        end
        
        # Process chunk with SIMD
        @inbounds @simd for i in 1:chunk_len
            # Stage 1: Initial combinations
            t1 = chunks[1][i] + chunks[5][i]
            t2 = chunks[1][i] - chunks[5][i]
            t3 = chunks[3][i] + chunks[7][i]
            t4 = -im * (chunks[3][i] - chunks[7][i])
            
            t5 = chunks[2][i] + chunks[6][i]
            t6 = chunks[2][i] - chunks[6][i]
            t7 = chunks[4][i] + chunks[8][i]
            t8 = -im * (chunks[4][i] - chunks[8][i])
            
            # Stage 2: Final combinations
            idx = base + i - 1
            y[idx] = t1 + t3
            y[idx + s] = t2 + INV_SQRT2 * (1-im) * (t6 + t8)
            y[idx + 2s] = t5 + t7
            y[idx + 3s] = -im * (t2 - INV_SQRT2 * (1+im) * (t6 - t8))
            y[idx + 4s] = t1 - t3
            y[idx + 5s] = t2 - INV_SQRT2 * (1-im) * (t6 + t8)
            y[idx + 6s] = t5 - t7
            y[idx + 7s] = im * (t2 - INV_SQRT2 * (1+im) * (t6 - t8))
        end
    end
end

"""
General FFT processing function that selects optimal parameters
"""
function process_fft!(
    y::AbstractVector{Complex{T}}, 
    x::AbstractVector{Complex{T}},
    radix::Int;
    cache_info::Union{CacheInfo, Nothing} = nothing
) where T <: AbstractFloat
    # Auto-detect cache parameters if not provided
    if isnothing(cache_info)
        cache_info = detect_cache_info()
    end
    
    # Calculate optimal chunk sizes
    chunk_params = calculate_chunk_sizes(cache_info, T)
    
    # Calculate optimal stride to avoid cache conflicts
    stride = get_optimal_stride(chunk_params.chunk_size, cache_info)
    
    n = length(x)
    s = n ÷ radix
    
    # Select appropriate kernel based on radix
    if radix == 4
        radix4_kernel_optimized!(y, x, s, chunk_params, stride)
    elseif radix == 8
        radix8_kernel_optimized!(y, x, s, chunk_params, stride)
    else
        throw(ArgumentError("Unsupported radix: $radix. Use 4 or 8."))
    end
    
    return y
end

"""
Calculate optimal stride to minimize cache conflicts
"""
function get_optimal_stride(chunk_size::Int, cache_info::CacheInfo)
    line_size = cache_info.l1d_linesize
    associativity = cache_info.l1d_associativity
    
    # Choose stride that's coprime with cache associativity
    base_stride = max(chunk_size ÷ associativity, line_size)
    return nextprime(base_stride)
end

# Benchmark utilities
using BenchmarkTools

function benchmark_radix_performance(
    sizes::Vector{Int} = [2^10, 2^12, 2^14, 2^16],
    radixes::Vector{Int} = [4, 8]
)
    cache_info = detect_cache_info()
    println("Cache Parameters:")
    println("L1D: $(cache_info.l1d_size) bytes, line size: $(cache_info.l1d_linesize)")
    println("L2: $(cache_info.l2_size) bytes")
    println("Vector size: $(cache_info.vector_size) bytes\n")
    
    for n in sizes
        println("Size: $n")
        x = [Complex{Float64}(rand(), rand()) for _ in 1:n]
        y = similar(x)
        
        for radix in radixes
            println("Radix-$radix:")
            b = @benchmark process_fft!($y, $x, $radix, cache_info=$cache_info)
            println("  Min time: $(minimum(b.times)) ns")
            println("  Mean time: $(mean(b.times)) ns")
            println("  Memory: $(b.memory) bytes")
        end
        println()
    end
end

export process_fft!, benchmark_radix_performance

end # module


