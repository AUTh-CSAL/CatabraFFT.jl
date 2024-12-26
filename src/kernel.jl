using AbstractFFTs, Primes 

include("radix_plan.jl")
include("radix_exec.jl")
include("mixed_radix.jl")
include("prime.jl")



using .Radix_Plan

if !@isdefined(F_cache)
    const F_cache = Dict{Tuple{Int, DataType}, Function}() 
end

struct Spell{T} <: AbstractFFTs.Plan{T} # Catabra's Plan
    n::Int
    mixed::Union{MixedRadixFFT{T}, Nothing}
    radixplan::Union{Vector{RadixPlan}, Nothing}
    ENCHANT::Bool
    pinv::Union{Spell{T}, Nothing}
end

if !@isdefined(spell_cache)
    const spell_cache = Dict{Tuple{Int, DataType}, Spell}()
end

@inline function generate_and_cache_fft!(n::Int, ::Type{T})::Function where {T <: AbstractFloat}
    key = (n, T)
    haskey(F_cache, key) && return F_cache[key]

    fft_func = if n == 1
        (y, x) -> (y .= x)
    elseif is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)
        call_radix_families(n, T)
    elseif isprime(n)
        generate_prime_fft_raders(n, T)
    else
        p, m = find_closest_factors(n)
        plan = MixedRadixFFT(p, m, T)
        generate_formulation_fft(plan, T)
    end

    # Cache-in
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
    show_function = false
    ivdep = false

    family_func = if is_power_of(n,2)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_2_plan(n, T), show_function, ivdep)
        elseif is_power_of(n, 3)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_3_plan(n, T), show_function, ivdep)
        elseif is_power_of(n, 5)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_plan(n, 5, T), show_function, ivdep)
        elseif is_power_of(n, 7)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_radix_plan(n, 7, T), show_function, ivdep)
    end

    return family_func
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

# Update the recursive_F function to use the new generator
function recursive_F(n::Int, ::Type{T})::Function where {T<:AbstractFloat}
    haskey(F_cache, n) && F_cache[n]

    fft_func = generate_and_cache_fft!(n, T)
    return fft_func
end
