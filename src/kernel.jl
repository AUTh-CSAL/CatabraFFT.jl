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
const COMPILED_FFT_EXPRS = Dict{Type{<:Spell}, Expr}()

function empty_kernel_cache()
    empty!(COMPILED_FFT_EXPRS)
end

"""
Generated function that produces optimized FFT kernel at compile time.
No world age issues, no invokelatest, just pure compiled performance.
"""
@generated function execute_fft!(spell::Spell{T,N,DECOMP,FLAG_VAL}, 
                                 y::AbstractVector{Complex{T}}, 
                                 x::AbstractVector{Complex{T}}) where {T,N,DECOMP,FLAG_VAL}
    # Generate constants dictionary at compile time
    constants_dict = RadixGenerator.generate_local_constants_dict(N, T)
    
    spell_type = Spell{T,N,DECOMP,FLAG_VAL}
    
    if haskey(COMPILED_FFT_EXPRS, spell_type)
        kernel_expr = COMPILED_FFT_EXPRS[spell_type]
    else
        # Generate optimized radix kernel expression
        kernel_expr = GenerateKernelExpr(N, T, FLAG(FLAG_VAL))
        COMPILED_FFT_EXPRS[spell_type] = kernel_expr
    end
    
    # Substitute constants with literal values
    substituted_kernel = Radix_Execute.substitute_constants_in_expr(kernel_expr, constants_dict)
    
    return quote
        @inline @fastmath @inbounds begin
            $substituted_kernel
        end
        nothing
    end
end

# Generate optimized FFT function at compile time
@inline function GenerateKernelExpr(n::Int, ::Type{T}, flag::FLAG)::Expr where {T <: AbstractFloat}
    spell_type = Spell{T, n, Tuple{}, Int(flag)}
    haskey(COMPILED_FFT_EXPRS, spell_type) && return COMPILED_FFT_EXPRS[spell_type]
    
    # Generate function based on size
    fft_func = if n == 1
        quote
            y[1] = x[1]
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

# Generate radix FFT function with compile-time optimizations
function generate_radix_fft_function(n::Int, ::Type{T}, flag::FLAG)::Expr where {T<:AbstractFloat}
    @assert is_power_of(n, 2)
    
    if flag >= ENCHANT
        plans = Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T)
        return Radix_Execute.return_best_static_linear_expr(plans, true)
    else
        plan = Radix_Plan.create_std_radix_plan(n, [8,4,2], T)
        return Radix_Execute.generate_mat_execute_function!(plan,true)
    end
end

# Direct kernel execution - no world age issues
@inline function fft_kernel_direct!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int) where T <: AbstractFloat
    spell = Spell(T, n, Tuple{}, NO_FLAG)
    execute_fft!(spell, y, x)
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