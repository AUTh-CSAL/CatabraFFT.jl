using AbstractFFTs, Primes 

include("radix_factory.jl")
include("radix_plan.jl")
include("radix_exec.jl")
include("mixed_radix.jl")
include("prime.jl")
include("spells.jl")
include("helper_tools.jl")

export Radix_Plan, RadixGenerator, Radix_Execute

# Store compiled functions directly - no dynamic module generation
const COMPILED_FFT_FUNCTIONS = Dict{Tuple{Int,Type,FLAG}, Function}()

function empty_kernel_cache()
    empty!(COMPILED_FFT_FUNCTIONS)
end

function get_cached_spell(n::Int, T::Type, flag::FLAG)
    key = (n, T, flag)
    haskey(COMPILED_FFT_FUNCTIONS, key) ? 
        Spell{T}(n, flag, COMPILED_FFT_FUNCTIONS[key]) : 
        nothing
end

function cache_spell!(spell::Spell{T}) where T
    key = (spell.n, spell.type, spell.flag)
    COMPILED_FFT_FUNCTIONS[key] = spell.fft_func
end

# Generate optimized FFT function at compile time
@inline function generate_optimized_fft_function(n::Int, ::Type{T}, flag::FLAG)::Function where {T <: AbstractFloat}
    # Check cache first
    key = (n, T, flag)
    haskey(COMPILED_FFT_FUNCTIONS, key) && return COMPILED_FFT_FUNCTIONS[key]
    
    # Generate function based on size
    fft_func = if n == 1
        (y, x) -> copyto!(y, x)
    elseif is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7)
        generate_radix_fft_function(n, T, flag)
    else
        # Simple DFT fallback for non-power sizes
        generate_dft_function(n, T)
    end
    
    # Cache and return
    COMPILED_FFT_FUNCTIONS[key] = fft_func
    return fft_func
end

# Generate radix FFT function with compile-time optimization
function generate_radix_fft_function(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    @assert (is_power_of(n, 2) || is_power_of(n, 3) || is_power_of(n, 5) || is_power_of(n, 7))
    
    if is_power_of(n, 2)
        if flag >= ENCHANT
            plans = Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T)
            # Generate and benchmark all plans, return best
            return Radix_Execute.return_best_static_linear_function(plans, true)
        elseif flag >= MEASURE
            plans = Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T)
            return Radix_Execute.return_best_linear_function(plans, false, false)
        else
            # Standard case - single plan
            plan = Radix_Plan.create_std_radix_plan(n, [8,4,2], T)
            return Radix_Execute.generate_mat_execute_function!(plan, false)
        end
    elseif is_power_of(n, 3)
        plan = Radix_Plan.create_std_radix_plan(n, [9, 3], T)
        return Radix_Execute.generate_mat_execute_function!(plan, false)
    elseif is_power_of(n, 5)
        plan = Radix_Plan.create_std_radix_plan(n, [5], T)
        return Radix_Execute.generate_mat_execute_function!(plan, false)
    elseif is_power_of(n, 7)
        plan = Radix_Plan.create_std_radix_plan(n, [7], T)
        return Radix_Execute.generate_mat_execute_function!(plan, false)
    else
        error("Unsupported radix")
    end
end

# Generate simple DFT function
function generate_dft_function(n::Int, ::Type{T})::Function where {T<:AbstractFloat}
    return function dft!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})
        @inbounds for k in 0:n-1
            y[k+1] = zero(Complex{T})
            for j in 0:n-1
                twiddle = exp(-2π * im * k * j / n)
                y[k+1] += x[j+1] * twiddle
            end
        end
        return nothing
    end
end

# Direct kernel execution - no world age issues
@inline function fft_kernel_direct!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int) where T <: AbstractFloat
    fft_func = generate_optimized_fft_function(n, T, NO_FLAG)
    fft_func(y, x)  # Direct call - no invokelatest
    return y
end

# Helper functions for real FFTs
function real_fft_kernel!(y::AbstractVector{Complex{T}}, x_work::AbstractVector{Complex{T}}, n::Int) where T
    fft_kernel_direct!(y, x_work, n)
end

function real_ifft_kernel!(y::AbstractVector{T}, x_work::AbstractVector{Complex{T}}, d::Int) where T
    temp = similar(x_work, d)
    fft_kernel_direct!(temp, x_work, length(x_work))
    for i in 1:d
        y[i] = real(temp[i])
    end
end