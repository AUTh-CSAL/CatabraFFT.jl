module CatabraFFT

# Opt out of precompilation to avoid method overwriting during dynamic code generation
#__precompile__(false)

include("kernel.jl")

using AbstractFFTs

import Base: show, *, convert, unsafe_convert, size, strides, ndims, pointer
import LinearAlgebra: mul!

# Non-mutating workspace that reuses preallocated memory
struct FFTWorkspace{T<:AbstractFloat}
    x_work::Vector{Complex{T}}
    function FFTWorkspace(n::Int, ::Type{T}) where {T<:AbstractFloat}
        new{T}(Vector{Complex{T}}(undef, n))
    end
end

# Thread-local workspace to avoid allocations in parallel code
const WORKSPACE = Dict{Tuple{Int, DataType}, FFTWorkspace}()
const WORKSPACE_LOCK = ReentrantLock()

# Get or create workspace for a given size - simplified
@inline function get_workspace(n::Int, ::Type{T})::FFTWorkspace where {T <: AbstractFloat}
    key = (n, T)
    workspace = get(WORKSPACE, key, nothing)
    if workspace !== nothing
        return workspace
    end
    lock(WORKSPACE_LOCK) do
        get!(WORKSPACE, key) do
            FFTWorkspace(n, T)
        end
    end
end

"""
Clear all internal caches used in CatabraFFT.

This empties caches like WORKSPACE and the kernel cache, which store preallocated
workspaces and cached FFT computations, to free up memory.

# Example
empty_cache()
"""
function empty_cache()
    empty!(WORKSPACE)
    empty_kernel_cache()
end

# in-place manipulation of given signal
@inline function fft!(x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    n = length(x)
    fft_kernel_direct!(x, x, n)
    return x
end

"""
    fft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}

Compute the 1-dimensional C2C Fast Fourier Transform (FFT) of the input vector.

# Arguments
- `x`: Input complex vector to be transformed

# Returns
- A vector containing the Fourier transform of the input

# Example
X = fft(x)
"""
@inline function fft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    fft_kernel_direct!(y, workspace.x_work, n)
    return y
end

"""
    ifft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}

Compute the 1-dimensional C2C Inverse Fast Fourier Transform (IFFT) of the input vector.

# Arguments
- `X`: Input complex vector to be inversely transformed

# Returns
- A vector containing the reverse Fourier transform of the input

# Example
x = ifft(X)
"""
@inline function ifft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    
    # IFFT using the FFT with complex conjugate and normalization
    conj!(workspace.x_work)
    fft_kernel_direct!(y, workspace.x_work, n)
    conj!(y)
    y ./= n
    
    return y
end

@inline function bfft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    
    # IFFT using the FFT with complex conjugate and NOT normalization (BFFT)
    conj!(workspace.x_work)
    fft_kernel_direct!(y, workspace.x_work, n)
    conj!(y)
    
    return y
end

"""
    plan_fft(x::AbstractVector{Complex{T}}, region=1:1; flags::FLAG=NO_FLAG) where T<:AbstractFloat

Create a plan for computing FFT of a complex vector. Supports optimization flags:
- ENCHANT: Enable special (technical) optimizations (similar to FFTW's PATIENT)
- MEASURE: Checks out possible strategies and sticks with the quickest, similar to FFTW's MEASURE flag
- NO_FLAG: Default planning strategy

Returns a Spell object that encapsulates the FFT plan with the optimized function stored directly.

# Arguments
- `x`: Input vector to plan FFT for
- `region`: Dimensions to transform (default: 1:1)  
- `flags`: Planning flags for optimization level

# Returns
- A Spell object representing the FFT plan

# Example
```julia
x = rand(Complex{Float64}, 1024)
p = plan_fft(x, flags=ENCHANT)
X = p * x
```
"""
function plan_fft(x::AbstractVector{Complex{T}}, flags::FLAG) where T <: AbstractFloat
    n = length(x)
    
    # Determine decomposition based on flags and size
    decomp = determine_decomposition(n, T, flags)
    
    # Create spell with the appropriate type parameters
    spell = Spell(T, n, decomp, flags)
    
    # Pre-generate and cache the kernel expression if not already cached
    spell_type = typeof(spell)
    if !haskey(COMPILED_FFT_EXPRS, spell_type)
        kernel_expr = GenerateKernelExpr(n, T, flags)
        COMPILED_FFT_EXPRS[spell_type] = kernel_expr
    end
    
    return spell
end

function AbstractFFTs.plan_fft(x::AbstractVector{Complex{T}}, region=1:1; flags::FLAG=NO_FLAG) where T <: AbstractFloat
    plan_fft(x, flags)
end

function AbstractFFTs.plan_bfft(x::AbstractVector{Complex{T}}, region=1:1) where T <: AbstractFloat
    plan_fft(x, NO_FLAG)
end

# Inverse plan caching - reuse existing spell structure
function AbstractFFTs.plan_inv(p::Spell{T}) where T
    if !isassigned(p.pinv)
        p.pinv[] = p  # For simplicity, same plan works (operation handled in execution)
    end
    return p.pinv[]
end

# Required * operation - direct execution of stored function
@inline function Base.:*(p::Spell{T,N,DECOMP,FLAG_VAL},
                        x::AbstractVector{Complex{T}}) where {T,N,DECOMP,FLAG_VAL} 
    #workspace = get_workspace(length(x), T)
    #copyto!(workspace.x_work, x)
    y = similar(x)
    # Execute the cached function directly - no invokelatest needed
    #execute_fft!(p, y, workspace.x_work)
    execute_fft!(p, y, x)
    return y
end

# Support for real FFTs (simplified)
function AbstractFFTs.plan_rfft(x::AbstractVector{T}, region=1:1) where T<:AbstractFloat
    n = length(x)
    # Create a spell for real FFT
    spell = Spell(T, n ÷ 2 + 1, (), NO_FLAG)
    return spell
end

function AbstractFFTs.plan_brfft(x::AbstractVector{Complex{T}}, d::Integer, region=1:1) where T<:AbstractFloat
    spell = Spell(T, d, (), NO_FLAG)
    spell.pinv[] = plan_rfft(zeros(T, d), region)
    return spell
end

# Adjoint support
AbstractFFTs.AdjointStyle(::Type{<:Spell}) = AbstractFFTs.FFTAdjointStyle()

function AbstractFFTs.adjoint_mul(y::AbstractVector{Complex{T}}, 
                                p::Spell{T}, 
                                x::AbstractVector{Complex{T}}) where T
    # For standard FFT, adjoint is same as inverse up to scaling
    copyto!(y, ifft(x))
end
    
AbstractFFTs.fftdims(p::Spell) = p.region
Base.size(p::Spell) = p.size

function (p::Spell{T})(x::AbstractVector{Complex{T}}) where T
    p * x
end

# Helper function to determine decomposition strategy
function determine_decomposition(n::Int, ::Type{T}, flags::FLAG) where T
    if n == 1
        return ()
    elseif is_power_of(n, 2)
        if flags >= ENCHANT
            # For ENCHANT, we use an optimized decomposition
            # This would be determined by the benchmarking in GenerateKernelExpr
            # For now, we'll use a standard decomposition
            return get_optimal_decomposition(n, T)
        else
            # Standard decomposition for power of 2
            return get_standard_decomposition(n)
        end
    else
        # Non-power-of-2 sizes
        return ()
    end
end

function get_optimal_decomposition(n::Int, ::Type{T}) where T
    # This would ideally be determined by benchmarking
    # For now, return a standard decomposition
    decomp = []
    remaining = n
    for radix in [8, 4, 2]
        while remaining % radix == 0
            push!(decomp, radix)
            remaining ÷= radix
        end
    end
    return Tuple(decomp)
end

function get_standard_decomposition(n::Int)
    decomp = []
    remaining = n
    for radix in [8, 4, 2]
        while remaining % radix == 0
            push!(decomp, radix)
            remaining ÷= radix
        end
    end
    return Tuple(decomp)
end

end