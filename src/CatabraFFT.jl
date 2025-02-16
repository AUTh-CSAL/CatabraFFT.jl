module CatabraFFT

include("kernel.jl")


using AbstractFFTs

import Base: show, *, convert, unsafe_convert, size, strides, ndims, pointer
import LinearAlgebra: mul!

# Non-mutating wrapper that reuses preallocated workspace
struct FFTWorkspace{T<:AbstractFloat}
    x_work::Vector{Complex{T}}
    function FFTWorkspace(n::Int, ::Type{T}) where {T<:AbstractFloat}
        new{T}(Vector{Complex{T}}(undef, n))
    end
end

# Thread-local workspace to avoid allocations in parallel code
const WORKSPACE = Dict{Tuple{Int, DataType}, FFTWorkspace}()

# Get or create workspace for a given size
@inline function get_workspace(n::Int, ::Type{T})::FFTWorkspace where {T <: AbstractFloat}
    key = (n, T)
    get!(WORKSPACE, key) do
        FFTWorkspace(n, T)
    end
end

"""
Clear all internal caches used in CatabraFFT.

This empties caches like WORKSPACE and F_cache, which store preallocated
workspaces and cached FFT computations, to free up memory.

# Example
empty_cache()
"""
function empty_cache()
    empty!(WORKSPACE)
    empty!(F_cache.forward)
    empty!(F_cache.backward)
end

# in-place manipulation of given signal
@inline function fft!(x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    fft_kernel!(x, x, CatabraFFT.NO_FLAG)
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
    fft_kernel!(y, workspace.x_work, CatabraFFT.NO_FLAG)
    return y
end

"""
    ifft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}

Compute the 1-dimensional C2C Inverse Fast Fourier Transform (IFFT) of the input vector.

# Arguments
- `X`: Input complex vector to be inversly transformed

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
    fft_kernel!(y, workspace.x_work, CatabraFFT.NO_FLAG)
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
    fft_kernel!(y, workspace.x_work, CatabraFFT.NO_FLAG)
    conj!(y)
    
    return y
end

"""
    plan_fft(x::AbstractVector{Complex{T}}, region=1:1; flags::FLAG=PLANNER_DEFAULT) where T<:AbstractFloat

Create a plan for computing FFT of a complex vector. Supports optimization flags:
- ENCHANT: Enable special (technical) optimizations (similar to FFTW's PATIENT)
- MEASURE: Checks out possible stategies and sticks with the quickers, similar to FFTW's MEASURE flag
- PLANNER_DEFAULT: Default planning strategy

Returns a Spell object that encapsulates the FFT plan.

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
#F_cache MONAD ?
#=
FFT --> Spell
|        ↓
↓        ↓
IFFT --> ISpell
=#
function plan_fft(x::AbstractVector{Complex{T}}, flags::FLAG)::Spell{T} where T <: AbstractFloat
    n = length(x)

    # Create spell first if flags are specified
    spell = Spell(n, T, flags) 
    key = (n, T, spell)
    
    # Check if we have a cached function for this configuration
    # We can return the spell directly since it matches the cache key
    haskey(F_cache, key) && spell 
    
    # Generate new function and cache it with its spell
    func = generate_and_cache_fft!(n, T, flags)
    
    # The generate_and_cache_fft! function will have cached the function with the spell
    # We can return the spell we created, as it's now associated with the function
    println("Spell: $spell")
    return get_spell_from_function(func)
end

function AbstractFFTs.plan_fft(x::AbstractVector{Complex{T}}, region=1:1; flags::FLAG=NO_FLAG) where T <: AbstractFloat
    plan_fft(x, flags)
end

function AbstractFFTs.plan_bfft(x::AbstractVector{Complex{T}}, region=1:1;) where T <: AbstractFloat
    p = Spell{T}(size(x), collect(region))
    p.pinv[] = plan_fft(x, region;)
    return p
end

# Inverse plan caching
function AbstractFFTs.plan_inv(p::Spell{T}) where T
    p.pinv === nothing && (p.pinv = Spell(p))
    return p.pinv
end

# Required mul! implementation
function mul!(y::AbstractVector{Complex{T}}, p::Spell, x::AbstractVector{Complex{T}}) where T
    fft_kernel!(y, x, p.flag)
    return y
end

# Required * operation
function Base.:*(p::Spell, x::AbstractVector{Complex{T}}) where T
    #y = similar(x)
    #mul!(y, p, x)
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x) # immutable x
    fft_kernel!(workspace.x_work, workspace.x_work, p.flag)
    return workspace.x_work
end

# Support for real FFTs
function AbstractFFTs.plan_rfft(x::AbstractVector{T}, region=1:1;) where T<:AbstractFloat
    n = length(x)
    Spell{T}((n ÷ 2 + 1,), collect(region))
end

function AbstractFFTs.plan_brfft(x::AbstractVector{Complex{T}}, d::Integer, region=1:1;) where T<:AbstractFloat
    p = Spell{T}((d,), collect(region))
    p.pinv[] = plan_rfft(zeros(T, d), region;)
    return p
end

# Adjoint support
AbstractFFTs.AdjointStyle(::Type{<:Spell}) = AbstractFFTs.FFTAdjointStyle()

function AbstractFFTs.adjoint_mul(y::AbstractVector{Complex{T}}, 
                                p::Spell{T}, 
                                x::AbstractVector{Complex{T}}) where T
    # For standard FFT, adjoint is same as inverse up to scaling
    plan_inv(p) * x
end
    

AbstractFFTs.fftdims(p::Spell) = p.region
Base.size(p::Spell) = p.size

function (p::Spell{T})(x::AbstractVector{Complex{T}}) where T
    fft(x)
end

end