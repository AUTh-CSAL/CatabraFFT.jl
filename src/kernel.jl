using AbstractFFTs, Primes 

include("radix_factory.jl")
include("radix_plan.jl")
include("radix_exec.jl")
include("mixed_radix.jl")
include("prime.jl")
include("spells.jl")
include("helper_tools.jl")

export Radix_Plan, RadixGenerator, Radix_Execute

const F_cache_lock = ReentrantLock()
const F_cache = Dict{Tuple{Int,Type,FLAG}, Spell}()

function get_cached_spell(n::Int, T::Type, flag::FLAG)
    key = (n, T, flag)
    lock(F_cache_lock) do
        get(F_cache, key, nothing)
    end
end

function cache_spell!(spell::Spell)
    key = (spell.n, spell.type, spell.flag)
    lock(F_cache_lock) do
        F_cache[key] = spell
    end
end

@inline function generate_and_cache_fft!(n::Int, ::Type{T}, flag::FLAG)::Function where {T <: AbstractFloat}
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
    return fft_func
end


@inline function fft_kernel!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, p::FLAG, n::Int) where T <: AbstractFloat
    fft_func = generate_and_cache_fft!(n, T, p) # NO_FLAG for fft(x) normal calls
    Base.invokelatest(fft_func, y, x) # FUNCTION EXECUTION
    #fft_func(y, x) # FUNCTION EXECUTION
    return y
end

function call_radix_families(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    @assert (is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)) "n: $n is not divisible by 2, 3, 5, or 7"
    show_function = true

    if is_power_of(n, 2)
        Radix_Execute.return_best_static_linear_function(Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T), show_function)
    end
    
    #=
    ivdep = false
    family_func = if flag >= ENCHANT
        if is_power_of(n, 2)
            Radix_Execute.return_best_static_linear_function(Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T), show_function)
        end
        elseif flag >= MEASURE
        #ivdep = flag >= ENCHANT ? true : false
        if is_power_of(n,2)
            #Radix_Execute.return_best_factorized_function(Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T), show_function, ivdep)
            Radix_Execute.return_best_linear_function(Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T), show_function, ivdep)
            #Radix_Execute.show_function_contents(CatabraFFT.Radix_Execute.execute_fft_linear!, AbstractVector{Complex{Float64}}, AbstractVector{Complex{Float64}})
        elseif is_power_of(n, 3)
            Radix_Execute.return_best_linear_function(Radix_Plan.create_all_radix_plans(n, [9, 3], T), show_function, ivdep)
        elseif is_power_of(n, 5)
            Radix_Execute.return_best_linear_function(Radix_Plan.create_all_radix_plans(n, [5], T), show_function, ivdep)
        elseif is_power_of(n, 7)
            Radix_Execute.return_best_linear_function(Radix_Plan.create_all_radix_plans(n, [7], T), show_function, ivdep)
        end
    else # no_flag
        if is_power_of(n,2)
            #Radix_Execute.generate_linear_execute_function!(Radix_Plan.create_std_radix_plan(n, [8,4,2], T), show_function, ivdep)
            Radix_Execute.generate_linear_execute_function!(Radix_Plan.create_std_radix_plan(n, [8,4,2], T), show_function, ivdep)
        elseif is_power_of(n, 3)
            Radix_Execute.generate_linear_execute_function!(Radix_Plan.create_std_radix_plan(n, [9,3], T), show_function, ivdep)
        elseif is_power_of(n, 5)
            Radix_Execute.generate_linear_execute_function!(Radix_Plan.create_std_radix_plan(n, [5], T), show_function, ivdep)
        elseif is_power_of(n, 7)
            Radix_Execute.generate_linear_execute_function!(Radix_Plan.create_std_radix_plan(n, [7], T), show_function, ivdep)
        end
    end
    =#

    return family_func
end

# Update the recursive_F function to use the new generator
function recursive_F(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    haskey(F_cache, n) && F_cache[n]

    fft_func = generate_and_cache_fft!(n, T, flag)
    return fft_func
end
