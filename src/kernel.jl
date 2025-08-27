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
@generated function execute_fft!(spell::Spell{T,N,DECOMP,FLAG_VAL}, 
                                 y::AbstractVector{Complex{T}}, 
                                 x::AbstractVector{Complex{T}})::Expr where {T,N,DECOMP,FLAG_VAL}
    # This code runs at compile time!
    # Generates the entire FFT kernel as an expression
    
    spell_type = Spell{T,N,DECOMP,FLAG_VAL}
    
    if haskey(COMPILED_FFT_EXPRS, spell_type)
        kernel_expr = COMPILED_FFT_EXPRS[spell_type]
    else
        # Generate optimized radix kernel expression
        kernel_expr = GenerateKernelExpr(N, T, FLAG(FLAG_VAL))
        COMPILED_FFT_EXPRS[spell_type] = kernel_expr
    end
    
    return quote
        @inbounds begin
            $kernel_expr
        end
        nothing
    end
end

# Store compiled functions directly - no dynamic module generation
# Use Type{<:Spell} as keys for maximum performance and type-level optimization
const COMPILED_FFT_EXPRS = Dict{Type{<:Spell}, Expr}()

function empty_kernel_cache()
    empty!(COMPILED_FFT_EXPRS)
end

# Generate optimized FFT function at compile time
@inline function GenerateKernelExpr(n::Int, ::Type{T}, flag::FLAG)::Expr where {T <: AbstractFloat}
    # Use type-level caching for maximum performance
    spell_type = Spell{T, n, Tuple{}, Int(flag)}
    haskey(COMPILED_FFT_EXPRS, spell_type) && return COMPILED_FFT_EXPRS[spell_type]
    
    # Generate function based on size
    fft_func = if n == 1
        quote
            @inbounds y[1] = x[1]
            nothing
        end
    elseif is_power_of(n, 2)
        generate_radix_fft_function(n, T, flag)
    else
        error("Unsupported transform size: $n")
    end
    
    # Cache at type level and return
    COMPILED_FFT_EXPRS[spell_type] = fft_func
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
        else
            # For non-ENCHANT flags, use single plan approach
            plan = Radix_Plan.create_std_radix_plan(n, [8,4,2], T)
            return Radix_Execute.generate_mat_execute_function!(plan, false)
        end
    else
        error("Unsupported radix for size: $n")
    end
end

# Direct kernel execution - no world age issues
@inline function fft_kernel_direct!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int) where T <: AbstractFloat
    fft_func_expr = GenerateKernelExpr(n, T, NO_FLAG)
    # Create a temporary spell for execution
    spell = Spell(T, n, Tuple{}, NO_FLAG)
    execute_fft!(spell, y, x) # Direct call - no invokelatest
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