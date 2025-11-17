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
Kernels expect Vector{T} where complex numbers are stored as [real1, imag1, real2, imag2, ...]
"""
@generated function execute_fft_impl!(spell::Spell{T,N,DECOMP,FLAG_VAL},
                                      py::AbstractVector{T},
                                      x::AbstractVector{T}) where {T,N,DECOMP,FLAG_VAL}
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
        # Define y as alias to py for compatibility with kernel code
        y = py
        @fastmath @inbounds begin
            $substituted_kernel
        end
        nothing
    end
end

# Wrapper that handles reinterpretation
@inline function execute_fft!(spell::Spell{T,N,DECOMP,FLAG_VAL},
                              y::AbstractVector{Complex{T}},
                              x::AbstractVector{Complex{T}}) where {T,N,DECOMP,FLAG_VAL}
    # Reinterpret Complex{T} arrays as T arrays for kernel access
    # Complex numbers are stored as consecutive pairs: [real1, imag1, real2, imag2, ...]
    x_real = reinterpret(T, x)
    y_real = reinterpret(T, y)
    execute_fft_impl!(spell, y_real, x_real)
    return nothing
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
    @assert is_power_of(n, 2) "n must be a power of 2"

    if flag >= ENCHANT
        plans = Radix_Plan.create_all_radix_plans(n, subpowers_of_two(n), T)
        result = Radix_Execute.return_best_static_linear_expr(plans, true)
        # Store benchmark results for later analysis if needed
        # Access via result.benchmarks
        for bench in enumerate(result.benchmarks)
            println("Time: $(bench[2]) \n")
        end
        return result.expr
    else
        plan = Radix_Plan.create_std_radix_plan(n, [8,4,2], T)
        # Return just the kernel body, not a full function definition
        return Radix_Execute.GenerateMatrixExpr!(plan, false)
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
    @inbounds for i in 1:d
        y[i] = real(temp[i])
    end
end