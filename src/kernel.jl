using AbstractFFTs, Primes 

include("radix_factory.jl")
include("radix_plan.jl")
include("radix_exec.jl")
include("mixed_radix.jl")
include("prime.jl")
include("spells.jl")

using .Radix_Plan 

# BiMap (HashMap) struct for FFT caching
mutable struct FFTCacheMap{T}
    forward::Dict{Tuple{Int, DataType, Union{Nothing, Spell}}, Function}
    backward::Dict{Function, Tuple{Int, DataType, Union{Nothing, Spell}}}

    FFTCacheMap{T}() where T = new{T}(Dict(), Dict())
end

# Initialize the global cache if not already defined
if !@isdefined(F_cache)
    const F_cache = FFTCacheMap{Function}()
end

# BiMap operations
@inline function Base.haskey(cache::FFTCacheMap, key::Union{Tuple{Int, DataType, Union{Nothing, Spell}}, Function})
    if key isa Tuple
        haskey(cache.forward, key)
    else
        haskey(cache.backward, key)
    end
end

@inline function Base.getindex(cache::FFTCacheMap, key::Union{Tuple{Int, DataType, Union{Nothing, Spell}}, Function})
    if key isa Tuple
        cache.forward[key]
    else
        cache.backward[key]
    end
end

@inline function Base.setindex!(cache::FFTCacheMap, value::Union{Function, Tuple{Int, DataType, Union{Nothing, Spell}}}, 
                              key::Union{Tuple{Int, DataType, Union{Nothing, Spell}}, Function})
    if key isa Tuple
        cache.forward[key] = value
        cache.backward[value] = key
    else
        cache.backward[key] = value
        cache.forward[value] = key
    end
    value
end

# Lookup functions
@inline function get_spell_from_function(func::Function)::Union{Nothing, Spell}
    haskey(F_cache, func) ? F_cache[func][3] : nothing
end

@inline function get_cache_info_from_function(func::Function)::Union{Nothing, Tuple{Int, DataType, Union{Nothing, Spell}}}
    haskey(F_cache, func) ? F_cache[func] : nothing
end

@inline function get_function_from_spell(spell::Spell{T})::Function where T <: AbstractFloat
    # Iterate over the forward dictionary to find the function corresponding to the given spell
    for ((_, _, cache_spell), func) in F_cache.forward
        if cache_spell == spell
            return func
        end
    end
    error("Function not found for the given spell.")
end

@inline function generate_and_cache_fft!(n::Int, ::Type{T}, flag::FLAG)::Function where {T <: AbstractFloat}
    spell = Spell(n, T, flag)
    key = (n, T, spell)
    haskey(F_cache, key) && return F_cache[key] # Check out if these requieremts already have a cached-in solution

    fft_func = if n == 1
        (y, x) -> (y .= x)
    elseif is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)
        call_radix_families(n, T, flag)
    elseif isprime(n)
        generate_prime_fft_raders(n, T, flag)
    else
        p, m = find_closest_factors(n)
        plan = MixedRadixFFT(p, m, T, flag)
        generate_formulation_fft(plan, T)
    end

    # Cache-in
    F_cache[key] = fft_func
    return fft_func
end


@inline function fft_kernel!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    n = length(x)
    fft_func = generate_and_cache_fft!(n, T, NO_FLAG) # NO_FLAG for fft(x) normal calls
    fft_func(y, x) # FUNCTION EXECUTION
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

function call_radix_families(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    @assert (is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)) "n: $n is not divisible by 2, 3, 5, or 7"
    
    show_function = true

    ivdep = false

    function subpowers_of_two(N::Int)
    # Check if N is a power of two
    @assert N > 1 && (N & (N - 1)) == 0 "N must be a power of two greater than 1"
    
    # Generate the list of subpowers
    subpowers = Vector{Int}()
    while N >= 2
        push!(subpowers, N)
        N = div(N, 2)
    end
    return subpowers
end
    
    family_func = if flag >= MEASURE
        ivdep = flag >= ENCHANT ? true : false
        if is_power_of(n,2)
            #RadixGenerator.evaluate_fft_generated_module(n, T) # make radix_2_family shells available
            Radix_Execute.return_best_family_function(Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T), show_function, ivdep)
        elseif is_power_of(n, 3)
            Radix_Execute.return_best_family_function(Radix_Plan.create_all_radix_plans(n, [9, 3], T), show_function, ivdep)
        elseif is_power_of(n, 5)
            Radix_Execute.return_best_family_function(Radix_Plan.create_all_radix_plans(n, [5], T), show_function, ivdep)
        elseif is_power_of(n, 7)
            Radix_Execute.return_best_family_function(Radix_Plan.create_all_radix_plans(n, [7], T), show_function, ivdep)
        end
    else # no_flag
        if is_power_of(n,2)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_std_radix_plan(n, [8,4,2], T), show_function, ivdep)
        elseif is_power_of(n, 3)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_std_radix_plan(n, [9,3], T), show_function, ivdep)
        elseif is_power_of(n, 5)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_std_radix_plan(n, [5], T), show_function, ivdep)
        elseif is_power_of(n, 7)
            Radix_Execute.generate_safe_execute_function!(Radix_Plan.create_std_radix_plan(n, [7], T), show_function, ivdep)
        end
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
function recursive_F(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    haskey(F_cache, n) && F_cache[n]

    fft_func = generate_and_cache_fft!(n, T, flag)
    return fft_func
end
