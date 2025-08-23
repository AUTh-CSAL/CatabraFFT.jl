using AbstractFFTs, Primes 

include("radix_factory.jl")
include("radix_plan.jl")
include("radix_exec.jl")
include("mixed_radix.jl")
include("prime.jl")
include("spells.jl")
include("helper_tools.jl")

export Radix_Plan, RadixGenerator, Radix_Execute

# kernel.jl - The magic happens here

"""
Generated function that produces optimized FFT kernel at compile time.
No world age issues, no invokelatest, just pure compiled performance.
Example:

 C = CatabraFFT.plan_fft{x, CatabraFFT.ENCHANT); execute_fft!(C, y, x)
"""
@generated function execute_fft!(::Spell{T,N,DECOMP,FLAG_VAL}, 
                                 y::AbstractVector{Complex{T}}, 
                                 x::AbstractVector{Complex{T}})::Expr where {T,N,DECOMP,FLAG_VAL}
    # This code runs at compile time!
    # Generate the entire FFT kernel as an expression
    
    # Generate optimized radix kernel
    kernel_expr = GenerateKernelExpr(N, T, FLAG_VAL)
    
    return quote
        @inbounds begin
            $kernel_expr
        end
        nothing
    end
end

"""
Runtime execution for RadixPlan during benchmarking.
This is slower but allows testing different plans.
"""
function execute_fft!(plan::RadixPlan{T}, 
                     y::AbstractVector{Complex{T}}, 
                     x::AbstractVector{Complex{T}}) where T
    # Get the expression
    kernel_expr = Radix_Execute.generate_mat_execute_expr!(plan, true)
    
    # Create a function and execute it
    func = eval(quote
        function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @inbounds begin
                $kernel_expr
            end
            nothing
        end
    end)
    
    Base.invokelatest(func, y, x)
end

# Store compiled functions directly - no dynamic module generation
const COMPILED_FFT_EXPRS = Dict{Tuple{Int,Type,FLAG}, Expr}()

function empty_kernel_cache()
    empty!(COMPILED_FFT_EXPRS)
end

function get_cached_spell(n::Int, T::Type, flag::FLAG)
    key = (n, T, flag)
    haskey(COMPILED_FFT_EXPRS, key) ? 
        Spell{T}(n, flag, COMPILED_FFT_EXPRS[key]) : 
        nothing
end

function cache_spell!(spell::Spell{T}) where T
    key = (spell.n, spell.type, spell.flag)
    COMPILED_FFT_EXPRS[key] = spell.fft_func
end

# Generate optimized FFT function at compile time
@inline function GenerateKernelExpr(n::Int, ::Type{T}, flag::FLAG)::Expr where {T <: AbstractFloat}
    # Check cache first
    key = (n, T, flag)
    haskey(COMPILED_FFT_EXPRS, key) && return COMPILED_FFT_EXPRS[key]
    
    # Generate function based on size
    fft_func = if n == 1
        return quote
            @inbounds y[1] = x[1]
            nothing
        end
    elseif is_power_of(n, 2)
        generate_radix_fft_function(n, T, flag)
    end
    
    # Cache and return
    COMPILED_FFT_EXPRS[key] = fft_func
    return fft_func
end

# Generate radix FFT function with compile-time optimizations - saturated twiddles
function generate_radix_fft_function(n::Int, ::Type{T}, flag::FLAG)::Expr where {T<:AbstractFloat}
    @assert is_power_of(n, 2)
    
    if is_power_of(n, 2)
        if flag >= ENCHANT
            plans = Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T)
            # Returns BODY expr; perfect to splice later
            return Radix_Execute.return_best_static_linear_expr(plans, true)
        end
    #=
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
        =#
        error("Unsupported radix")
    end
end

# Direct kernel execution - no world age issues
@inline function fft_kernel_direct!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int) where T <: AbstractFloat
    fft_func_expr = GenerateKernelExpr(n, T, NO_FLAG)
    execute_fft!(Spell(T, N, Tuple(N)), y, x) # Direct call - no invokelatest
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