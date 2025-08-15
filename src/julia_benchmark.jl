module CatabraFFT

using AbstractFFTs
using LinearAlgebra
using Primes
import Base: *, show

# ==================== Helper Functions (Must be defined FIRST) ====================

function find_optimal_factors(n::Int)
    if isprime(n)
        return 1, n
    end

    # Try to find factors close to sqrt(n)
    sqrtn = isqrt(n)
    for p in sqrtn:-1:2
        if n % p == 0
            return p, n ÷ p
        end
    end
    return 1, n
end

function primitive_root(p::Int)
    if p == 2
        return 1
    end

    # Find prime factors of p-1
    factors = keys(factor(p-1))

    for g in 2:(p-1)
        is_root = true
        for q in factors
            if powermod(g, (p-1)÷q, p) == 1
                is_root = false
                break
            end
        end
        if is_root
            return g
        end
    end
    error("No primitive root found for $p")
end

function factor(n::Int)
    factors = Dict{Int,Int}()
    d = 2
    while d * d <= n
        while n % d == 0
            factors[d] = get(factors, d, 0) + 1
            n ÷= d
        end
        d += 1
    end
    if n > 1
        factors[n] = get(factors, n, 0) + 1
    end
    return factors
end

# ==================== Kernel Generation Functions ====================

# Generate optimized radix-2 kernel at compile time
function generate_radix2_kernel(N::Int, ::Type{T}) where T
    if N == 2
        return quote
            @inbounds begin
                a = x[1]
                b = x[2]
                y[1] = a + b
                y[2] = a - b
            end
            nothing
        end
    elseif N == 4
        return generate_radix4_kernel_expr(T)
    elseif N == 8
        return generate_radix8_kernel_expr(T)
    else
        # Stockham auto-sort for larger powers of 2
        return generate_stockham_kernel(N, T)
    end
end

# Optimized radix-4 kernel
function generate_radix4_kernel_expr(::Type{T}) where T
    return quote
        @inbounds begin
            # Load with bit-reversed addressing
            x0, x1, x2, x3 = x[1], x[2], x[3], x[4]

            # First butterfly layer
            a0 = x0 + x2
            a1 = x0 - x2
            a2 = x1 + x3
            a3 = (x1 - x3) * im

            # Second butterfly layer
            y[1] = a0 + a2
            y[2] = a1 - a3
            y[3] = a0 - a2
            y[4] = a1 + a3
        end
        nothing
    end
end

# Optimized radix-8 kernel with precomputed twiddles
function generate_radix8_kernel_expr(::Type{T}) where T
    invsqrt2 = T(1/sqrt(2))

    return quote
        @inbounds begin
            local invsqrt2 = $invsqrt2

            # Load inputs
            x0, x1, x2, x3 = x[1], x[2], x[3], x[4]
            x4, x5, x6, x7 = x[5], x[6], x[7], x[8]

            # Stage 1: Radix-2 butterflies
            s1_0 = x0 + x4
            s1_1 = x0 - x4
            s1_2 = x1 + x5
            s1_3 = x1 - x5
            s1_4 = x2 + x6
            s1_5 = x2 - x6
            s1_6 = x3 + x7
            s1_7 = x3 - x7

            # Apply twiddles
            t3 = s1_3 * im
            t5 = Complex{$T}(invsqrt2 * (real(s1_5) + imag(s1_5)),
                            invsqrt2 * (imag(s1_5) - real(s1_5)))
            t7 = s1_7 * -im

            # Stage 2: Radix-2 butterflies
            s2_0 = s1_0 + s1_4
            s2_1 = s1_1 + t5
            s2_2 = s1_2 + s1_6
            s2_3 = t3 + t7
            s2_4 = s1_0 - s1_4
            s2_5 = s1_1 - t5
            s2_6 = s1_2 - s1_6
            s2_7 = t3 - t7

            # Stage 3: Final butterflies
            y[1] = s2_0 + s2_2
            y[2] = s2_1 + s2_3
            y[3] = s2_4 + s2_6 * im
            y[4] = s2_5 + s2_7 * im
            y[5] = s2_0 - s2_2
            y[6] = s2_1 - s2_3
            y[7] = s2_4 - s2_6 * im
            y[8] = s2_5 - s2_7 * im
        end
        nothing
    end
end

# Generate Stockham auto-sort FFT for larger sizes
function generate_stockham_kernel(N::Int, ::Type{T}) where T
    @assert ispow2(N) && N >= 16

    # Generate nested loops with compile-time known bounds
    stages = Int(log2(N))

    expr = quote
        @inbounds begin
            # Use y as output directly for first stage
            local src = x
            local dst = y
            local temp_buffer = Vector{Complex{$T}}(undef, $N)
        end
    end

    # Build stages
    stage_exprs = []
    for stage in 1:stages
        m = 1 << stage
        m2 = m >> 1

        stage_expr = quote
            let m = $m, m2 = $m2
                stride = $N ÷ m
                for j in 0:(m2-1)
                    # Precompute twiddle
                    w = cispi($T(-2) * j / m)

                    @simd ivdep for k in 0:(stride-1)
                        idx1 = k * m + j + 1
                        idx2 = idx1 + m2

                        a = src[idx1]
                        b = src[idx2] * w

                        dst[k * m + j + 1] = a + b
                        dst[k * m + j + m2 + 1] = a - b
                    end
                end
                # Swap buffers
                if dst === y
                    src = y
                    dst = temp_buffer
                else
                    src = temp_buffer
                    dst = y
                end
            end
        end
        push!(stage_exprs, stage_expr)
    end

    # Add stages to main expression
    append!(expr.args[2].args, stage_exprs)

    # Ensure result is in y
    push!(expr.args[2].args, quote
        if dst !== y
            copyto!(y, temp_buffer)
        end
    end)

    push!(expr.args[2].args, :nothing)
    return expr
end

# Small Prime FFT (Rader's Algorithm)
function generate_small_prime_kernel(N::Int, ::Type{T}) where T
    @assert isprime(N) && N < 64

    # Find primitive root at compile time
    g = primitive_root(N)
    g_inv = powermod(g, N-2, N)

    # Precompute permutation indices
    perm = [(powermod(g, i, N) - 1) for i in 0:(N-2)]
    iperm = [(powermod(g_inv, i, N) - 1) for i in 0:(N-2)]

    # Precompute twiddle factors
    twiddles = [cispi(T(-2) * (iperm[i] + 1) / N) for i in 0:(N-2)]

    # Generate kernel expression
    return quote
        @inbounds begin
            # DC component
            dc = x[1]
            sum_val = dc
            @simd for i in 2:$N
                sum_val += x[i]
            end
            y[1] = sum_val

            # Rader's convolution via direct computation for small primes
            temp = Vector{Complex{$T}}(undef, $(N-1))

            # Permute input
            $([:(temp[$i+1] = x[$(perm[i]+2)]) for i in 0:(N-2)]...)

            # Direct convolution with precomputed twiddles
            for k in 1:$(N-1)
                conv_sum = zero(Complex{$T})
                for j in 1:$(N-1)
                    idx = mod1(k - j + 1, $(N-1))
                    conv_sum += temp[j] * $twiddles[idx]
                end
                temp[k] = conv_sum
            end

            # Inverse permute output
            $([:(y[$(iperm[i]+2)] = dc + temp[$i+1]) for i in 0:(N-2)]...)
        end
        nothing
    end
end

# Mixed Radix FFT
function generate_mixed_radix_kernel(p::Int, m::Int, ::Type{T}) where T
    N = p * m

    # For mixed radix, we'll use a simpler approach that doesn't require recursive calls
    return quote
        @inbounds begin
            # Direct DFT implementation for mixed radix
            # This avoids recursive calls which cause world age issues

            # Compute DFT directly using Cooley-Tukey decomposition
            for k in 0:$(N-1)
                sum_val = zero(Complex{$T})
                for n in 0:$(N-1)
                    # Decompose indices
                    n1 = n % $p
                    n2 = n ÷ $p
                    k1 = k % $m
                    k2 = k ÷ $m

                    # Compute twiddle factors
                    w = cispi($T(-2) * (k1*n1/$p + k2*n2/$m + k1*n2/$N))
                    sum_val += x[n + 1] * w
                end
                y[k + 1] = sum_val
            end
        end
        nothing
    end
end

# ==================== Core Types ====================

# Type-stable FFT plan with compile-time size information
struct FFTPlan{N, T<:AbstractFloat, F} <: AbstractFFTs.Plan{Complex{T}}
    forward::F
    workspace::Vector{Complex{T}}

    function FFTPlan{N, T}() where {N, T<:AbstractFloat}
        workspace = Vector{Complex{T}}(undef, N)
        forward = get_fft_functor(Val(N), T)
        new{N, T, typeof(forward)}(forward, workspace)
    end
end

# Functor pattern for type-stable function dispatch
abstract type AbstractFFTFunctor{N, T} end

struct Radix2Functor{N, T} <: AbstractFFTFunctor{N, T} end
struct PrimeFunctor{N, T} <: AbstractFFTFunctor{N, T} end
struct MixedRadixFunctor{N, P, M, T} <: AbstractFFTFunctor{N, T}
    function MixedRadixFunctor{N, P, M, T}() where {N, P, M, T}
        @assert N == P * M "N must equal P * M"
        new{N, P, M, T}()
    end
end
struct DirectDFTFunctor{N, T} <: AbstractFFTFunctor{N, T} end

# ==================== Main FFT Implementation ====================

# Use non-generated functions for each size to avoid world age issues
function apply_fft_2!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T
    @inbounds begin
        a = x[1]
        b = x[2]
        y[1] = a + b
        y[2] = a - b
    end
    nothing
end

function apply_fft_4!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T
    @inbounds begin
        x0, x1, x2, x3 = x[1], x[2], x[3], x[4]

        a0 = x0 + x2
        a1 = x0 - x2
        a2 = x1 + x3
        a3 = (x1 - x3) * im

        y[1] = a0 + a2
        y[2] = a1 - a3
        y[3] = a0 - a2
        y[4] = a1 + a3
    end
    nothing
end

function apply_fft_8!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T
    invsqrt2 = T(1/sqrt(2))
    @inbounds begin
        x0, x1, x2, x3 = x[1], x[2], x[3], x[4]
        x4, x5, x6, x7 = x[5], x[6], x[7], x[8]

        s1_0 = x0 + x4
        s1_1 = x0 - x4
        s1_2 = x1 + x5
        s1_3 = x1 - x5
        s1_4 = x2 + x6
        s1_5 = x2 - x6
        s1_6 = x3 + x7
        s1_7 = x3 - x7

        t3 = s1_3 * im
        t5 = Complex{T}(invsqrt2 * (real(s1_5) + imag(s1_5)),
                        invsqrt2 * (imag(s1_5) - real(s1_5)))
        t7 = s1_7 * -im

        s2_0 = s1_0 + s1_4
        s2_1 = s1_1 + t5
        s2_2 = s1_2 + s1_6
        s2_3 = t3 + t7
        s2_4 = s1_0 - s1_4
        s2_5 = s1_1 - t5
        s2_6 = s1_2 - s1_6
        s2_7 = t3 - t7

        y[1] = s2_0 + s2_2
        y[2] = s2_1 + s2_3
        y[3] = s2_4 + s2_6 * im
        y[4] = s2_5 + s2_7 * im
        y[5] = s2_0 - s2_2
        y[6] = s2_1 - s2_3
        y[7] = s2_4 - s2_6 * im
        y[8] = s2_5 - s2_7 * im
    end
    nothing
end

# Generic Stockham FFT for powers of 2
function apply_fft_pow2!(N::Int, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T
    @assert ispow2(N) && N >= 16
    stages = Int(log2(N))

    # Copy input to output
    copyto!(y, x)
    temp = similar(y)

    src = y
    dst = temp

    @inbounds for stage in 1:stages
        m = 1 << stage
        m2 = m >> 1
        stride = N ÷ m

        for j in 0:(m2-1)
            w = cispi(T(-2) * j / m)

            @simd ivdep for k in 0:(stride-1)
                idx1 = k * m + j + 1
                idx2 = idx1 + m2

                a = src[idx1]
                b = src[idx2] * w

                dst[k * m + j + 1] = a + b
                dst[k * m + j + m2 + 1] = a - b
            end
        end

        src, dst = dst, src
    end

    # Ensure result is in y
    if src !== y
        copyto!(y, src)
    end

    nothing
end

# Direct DFT for arbitrary sizes (fallback)
function apply_dft!(N::Int, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T
    @inbounds for k in 0:(N-1)
        sum_val = zero(Complex{T})
        for n in 0:(N-1)
            w = cispi(T(-2) * k * n / N)
            sum_val += x[n + 1] * w
        end
        y[k + 1] = sum_val
    end
    nothing
end

# Main dispatcher using Val types
function apply_fft!(::Val{N}, y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where {N, T}
    if N == 1
        @inbounds y[1] = x[1]
    elseif N == 2
        apply_fft_2!(y, x)
    elseif N == 4
        apply_fft_4!(y, x)
    elseif N == 8
        apply_fft_8!(y, x)
    elseif ispow2(N)
        apply_fft_pow2!(N, y, x)
    else
        apply_dft!(N, y, x)
    end
    nothing
end

# Functor application
@inline function (f::Radix2Functor{N, T})(y, x) where {N, T}
    apply_fft!(Val(N), y, x)
end

@inline function (f::PrimeFunctor{N, T})(y, x) where {N, T}
    apply_fft!(Val(N), y, x)
end

@inline function (f::MixedRadixFunctor{N, P, M, T})(y, x) where {N, P, M, T}
    apply_fft!(Val(N), y, x)
end

@inline function (f::DirectDFTFunctor{N, T})(y, x) where {N, T}
    apply_dft!(N, y, x)
end

# Get appropriate functor based on size
function get_fft_functor(::Val{N}, ::Type{T}) where {N, T}
    if ispow2(N) && N <= 8
        return Radix2Functor{N, T}()
    elseif ispow2(N)
        return Radix2Functor{N, T}()
    elseif isprime(N) && N < 64
        return PrimeFunctor{N, T}()
    else
        p, m = find_optimal_factors(N)
        if p > 1 && m > 1
            return MixedRadixFunctor{N, p, m, T}()
        else
            return DirectDFTFunctor{N, T}()
        end
    end
end

# ==================== Public API ====================

"""
    fft(x::AbstractVector{Complex{T}}) where T

Compute the FFT of x using optimized kernels.
"""
function fft(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    N = length(x)
    y = similar(x)
    apply_fft!(Val(N), y, x)
    return y
end

"""
    fft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T

Compute the FFT of x into y using optimized kernels.
"""
function fft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    N = length(x)
    @assert length(y) == N "Output length must match input length"
    apply_fft!(Val(N), y, x)
    return y
end

"""
    plan_fft(x::AbstractVector{Complex{T}}, flags=nothing) where T

Create an FFT plan for the given input size.
"""
function plan_fft(x::AbstractVector{Complex{T}}, flags=nothing) where T<:AbstractFloat
    N = length(x)
    return FFTPlan{N, T}()
end

# AbstractFFTs interface
function Base.:*(p::FFTPlan{N, T}, x::AbstractVector{Complex{T}}) where {N, T}
    y = similar(x)
    p.forward(y, x)
    return y
end

# Inverse FFT support
function apply_ifft!(::Val{N}, y::AbstractVector{Complex{T}},
                    x::AbstractVector{Complex{T}}) where {N, T}
    # Conjugate input
    x_conj = conj.(x)
    apply_fft!(Val(N), y, x_conj)
    # Conjugate output and scale
    @inbounds @simd for i in 1:N
        y[i] = conj(y[i]) / N
    end
    nothing
end

function ifft(x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    N = length(x)
    y = similar(x)
    apply_ifft!(Val(N), y, x)
    return y
end

function ifft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T<:AbstractFloat
    N = length(x)
    @assert length(y) == N "Output length must match input length"
    apply_ifft!(Val(N), y, x)
    return y
end

# ==================== Benchmarking Utilities ====================

"""
    benchmark_fft(n::Int, ::Type{T}=Float64; iterations=1000) where T

Benchmark the FFT implementation for size n.
"""
function benchmark_fft(n::Int, ::Type{T}=Float64; iterations=1000) where T<:AbstractFloat
    x = randn(Complex{T}, n)
    y = similar(x)

    # Warmup
    fft!(y, x)

    # Benchmark
    times = Float64[]
    for _ in 1:iterations
        t = @elapsed fft!(y, x)
        push!(times, t)
    end

    mean_time = sum(times) / length(times)
    min_time = minimum(times)

    return (mean=mean_time, min=min_time, times=times)
end

# Export main functions
export fft, fft!, ifft, ifft!, plan_fft, benchmark_fft

end # module