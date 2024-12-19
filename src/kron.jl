using Base.GC, LinearAlgebra, LoopVectorization, Primes, RuntimeGeneratedFunctions

include("radix_plan.jl")
include("radix_exec.jl")

if !@isdefined(MixedRadixFFT)
struct MixedRadixFFT{T <: AbstractFloat}
    p::Int
    m::Int
    W::Matrix{Complex{T}}
    Fm::Function  # Row-wise FFT function
    Fp::Function  # Column-wise FFT function
end
end

if !@isdefined(plan_cache)
    const plan_cache = Dict{Tuple{Int, Int, DataType}, MixedRadixFFT}() 
end

if !@isdefined(F_cache)
    const F_cache = Dict{Tuple{Int, DataType}, Function}() 
end

@inline function generate_and_cache_fft!(n::Int, ::Type{T})::Function where {T <: AbstractFloat}
    key = (n, T)
    haskey(F_cache, key) && return F_cache[key]

    if n == 1
        fft_func = (y, x) -> (y .= x)
    elseif is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)
        fft_func = call_radix_families(n, T)
    elseif isprime(n)
        fft_func = generate_prime_fft_raders(n, T)
    else
        p, m = find_closest_factors(n)
        plan = MixedRadixFFT(p, m, T)
        fft_func = generate_formulation_fft(plan, T)
    end

    #Cache-in
    F_cache[key] = fft_func
    return fft_func
end

"""
    fft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}

Compute the 1-dimensional C2C Fast Fourier Transform (FFT).

# Arguments
- `y`: Complex vector to store the result
- `x`: Input complex vector to be transformed

# Returns
- A vector containing the Fourier transform of the input

# Example
 fft!(y,x)
"""
@inline function fft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    n = length(x)
    fft_func = generate_and_cache_fft!(n, T)
    fft_func(y, x)
    return y
end

function MixedRadixFFT(p::Int, m::Int, ::Type{T})::MixedRadixFFT where {T<:AbstractFloat}
    key = (p, m, T)
    haskey(plan_cache, key) && return plan_cache[key]

    W = D(p, m, T)
    Fm = recursive_F(m, T)
    Fp = recursive_F(p, T)
    plan_cache[key] = MixedRadixFFT{T}(p, m, W, Fm, Fp)
    MixedRadixFFT{T}(p, m, W, Fm, Fp)
end

function is_power_of(n::Int, p::Int)
    while n > 1
        if n % p != 0
            return false
        end
        n ÷= p
    end
    return true
end

function call_radix_families(n::Int, ::Type{T})::Function where {T<:AbstractFloat}
    @assert (is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)) "n: $n is not divisible by 2, 3, 5, or 7"

    key = (n, T)
    haskey(F_cache, key) && return F_cache[key]

    family_func = if is_power_of(n,2)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_2_plan(n, T))
        elseif is_power_of(n, 3)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_3_plan(n, T))
        elseif is_power_of(n, 5)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_plan(n, 5, T))
        elseif is_power_of(n, 7)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_plan(n, 7, T))
    end

    return family_func
end

function generate_prime_fft_raders(n::Int, ::Type{T})::Function where {T<:AbstractFloat}
    @assert isprime(n) "Input length must be prime for Rader's FFT"

    # Find primitive root
    function generator(p)
        for gen in 2:(p-1)
            powers = Set{Int}()
            current = 1
            valid = true
            for i in 1:(p-1)
                current = (current * gen) % p
                if current in powers
                    valid = false
                    break
                end
                push!(powers, current)
            end
            if valid
                perm = [powermod(gen, i, p) for i in 0:(p-2)]
                return gen, perm
            end
        end
        error("No generator found")
    end

    # Generate the generator and permutation sequence
    gen, gen_seq = generator(n)
    inv_gen = powermod(gen, n-2, n)
    inv_seq = [powermod(inv_gen, i, n) for i in 0:(n-2)]

    # Precompute twiddle factors
    ω = cispi(T(2/n))
    W = [ω^(-inv_seq[i]) for i in 1:(n-1)]

    # Get FFT for length n-1
    F = recursive_F(n-1, T)

    return function (y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where {T<:AbstractFloat}
        @assert length(x) == n && length(y) == n "Input and output vectors must have length n"

        # Preallocate two buffers
        buffer1 = similar(x, n-1)
        buffer2 = similar(x, n-1)

        fill!(buffer1, zero(ComplexF64))
        fill!(buffer2, zero(ComplexF64))

        x0 = x[1]

        @inbounds @simd for i in 1:(n-1)
            buffer1[i] = x[gen_seq[i] + 1]
        end

        y[1] = x0 + sum(buffer1)

        F(buffer2, buffer1)  # FFT of permuted input -> buffer2
        F(buffer1, W)        # FFT of twiddle factors -> buffer1

        # Multiply in frequency domain (reuse buffer1 or buffer2 as needed)
        @inbounds @simd for i in 1:(n-1)
            buffer2[i] *= buffer1[i]
        end

        # Inverse FFT (reuse buffer1 for the result)
        F(buffer1, conj.(buffer2))

        @inbounds @simd for i in 1:(n-1)
            buffer1[i] = conj(buffer1[i]) / (n-1)
        end

        # Step 5: Place remaining terms
        @inbounds @simd for j in 1:(n-1)
            y[inv_seq[j] + 1] = x0 + buffer1[j]
        end

        return nothing
    end
end

function return_sorted_prime_powers(n::Int)
    primes = [2,3,5,7] # Primes I have families of
    prime_powers = []

    for prime in primes
        pow = prime
        while pow <= n
            push!(prime_powers, pow)
            if pow > typemax(Int) ÷ prime #stack overflow protection
                break
            end
            pow *= prime
        end
    end

    # Insertion sort (descending order)
    sort!(prime_powers, rev=true)
    return prime_powers
end

#When p ≈ m fewer matrices are recomputed => better runtime.
# Special strided FFT kernels with lower radix rank for special computation of n
# => mixed-radix-(m,p) !!!
function find_closest_factors(n::Int, prime_powers_preference=true)
    if isprime(n)
        return 1, n
    end
    if prime_powers_preference
        prime_powers = return_sorted_prime_powers(n)

        for p in prime_powers
            if n % p == 0
                return p, div(n,p)
            end
        end
    end
    p = isqrt(n) # Start with p as the floor of sqrt(n)

    while n % p != 0 # Adjust p until it divides n evenly
        p -= 1
        if p == 1
            error("Unable to find non-prime factors for $n")
        end
    end
    return p, div(n, p)
end

@inline function D(p,m, ::Type{T})::Matrix where {T<:AbstractFloat}
  w = cispi.(T(-2/(p*m)) * collect(1:m-1))
  d = zeros(Complex{T},(p-1)*(m-1))

  @inbounds d[1:m-1] .= w

  @inbounds @simd for j in 2:p-1
        @views d[(j-1)*(m-1)+1:j*(m-1)] .= w .* view(d, (j-2)*(m-1)+1:(j-1)*(m-1))
  end

  return reshape(d, m-1, p-1)
end

# vec(transpose(F(m)*(W.*(Xmp*F(p))))) ≈ y
function generate_formulation_fft(plan::MixedRadixFFT, ::Type{T})::Function where {T<:AbstractFloat}
    p, m = plan.p, plan.m
    W = D(p,m, T)

    Ypm = Matrix{Complex{T}}(undef, p, m)
    Xpm = Matrix{Complex{T}}(undef, p, m)

    return @inline function (y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})

        Xpm .= reshape(x, p, m)
        Ypm .= zero(Complex{T})

        @inbounds @simd for i in 1:p
            plan.Fm(@view(Ypm[i, :]), @view(Xpm[i, :]))
        end

        @inbounds @simd for i in 1:p-1
            @inbounds @simd for j in 1:m-1
                Ypm[i + 1, j + 1] *= W[j,i]
            end
        end

        # Step 3: Transpose the result and apply Fp column-wise
        # (W^T (X_pm Fm))^T Fp
        @inbounds @simd for j in 1:m
            plan.Fp(@view(Xpm[:, j]), @view(Ypm[:, j]))
        end

        # Step 4: Write the output to the vector y
        y .= vec(transpose(Xpm))

        return nothing
    end
end

# Update the recursive_F function to use the new generator
function recursive_F(n::Int, ::Type{T})::Function where {T<:AbstractFloat}
    haskey(F_cache, n) && F_cache[n]

    fft_func = generate_and_cache_fft!(n, T)
    return fft_func
end

## NOTE
#=
The use of Base.invokelatest is necessary in your code because new functions (Fm and Fp) are being created dynamically at runtime in recursive_F. This causes Julia’s Just-In-Time (JIT) compiler to defer compilation, so Base.invokelatest ensures the latest version of these functions is used. However, Base.invokelatest introduces some overhead and is often best avoided in performance-sensitive code.
To eliminate the need for Base.invokelatest, you can consider these approaches:
Eager Compilation of Fm and Fp: Instead of dynamically creating functions during the FFT computation, you can precompute and cache these functions ahead of time for each required size. This approach removes the need for Base.invokelatest by ensuring the function is already compiled before being used in formulation!.
Precompute Plans: If the plan is created multiple times with the same parameters, consider caching the plan itself (including Fm and Fp) for each (p, m) pair so it doesn’t require dynamic function generation.
=#
