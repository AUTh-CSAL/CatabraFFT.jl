module CatabraFFT

include("kernel.jl")

import AbstractFFTs:Plan, ScaledPlan, fft, ifft, bfft, fft!, ifft!, bfft!,
                    plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
                    rfft, irfft, brfft, plan_ffft, plan_irfft, plan_brfft,
                    fftshift, ifftshift, rfft_output_size, brfft_output_size,
                    plan_inv, normalization

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
    empty!(F_cache)
    #empty!(spell_cache)
end

@inline function fft!(x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    fft_kernel!(y, x)
    return y
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
    fft_kernel!(y, workspace.x_work)
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
x = fft(X)
"""
@inline function ifft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    
    # IFFT using the FFT with complex conjugate and normalization
    conj!(workspace.x_work)
    fft_kernel!(y, workspace.x_work)
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
    fft_kernel!(y, workspace.x_work)
    conj!(y)
    
    return y
end

"""
    plan_fft(x::AbstractVector{Complex{T}}, flags::FLAG=PLANNER_DEFAULT) where {T<:AbstractFloat}

Create a plan for computing FFT of a complex vector.
"""
function plan_fft(x::AbstractVector{Complex{T}}, region; flags::FLAG=PLANNER_DEFAULT) where {T<:AbstractFloat}
    n = length(x)
    
    # Cache spell if ENCHANT is enabled
    if flags & ENCHANT != 0
        spell_cache[(n, T)] = spell
    end
    
    spell
end

function AbstractFFTs.plan_fft(x::AbstractVector{Complex{T}}, region=1:1; kws...) where T <: AbstractFloat
    p = Spell{T}(size(x), collect(region))
    p.pinv[] = plan_fft(x, region;)
    return p
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
    copyto!(y, fft(x))
    return y
end

# Required * operation
function Base.:*(p::Spell, x::AbstractVector{Complex{T}}) where T
    y = similar(x)
    mul!(y, p, x)
    return y
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