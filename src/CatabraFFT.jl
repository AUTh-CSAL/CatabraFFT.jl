module CatabraFFT

include("kron.jl")

__precompile__()
GC.gc()

# Non-mutating wrapper that reuses preallocated workspace
struct FFTWorkspace{T<:AbstractFloat}
    x_work::Vector{Complex{T}}
    function FFTWorkspace(n::Int, ::Type{T}) where {T<:AbstractFloat}
        new{T}(Vector{Complex{T}}(undef, n))
    end
end

# Thread-local workspace to avoid allocations in parallel code
const WORKSPACE = Dict{Tuple{Int, DataType}, FFTWorkspace}()
const F_cache = Dict{Tuple{Int, DataType}, Function}()  # Example cache for FFT computations

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
@inline function fft(x::AbstractVector{Complex{T}}, use_ivdep::Bool=false)::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    fft!(y, workspace.x_work, use_ivdep)
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
@inline function ifft(x::AbstractVector{Complex{T}}, use_ivdep::Bool=false)::AbstractVector{Complex{T}} where {T <: AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    
    # IFFT using the FFT with complex conjugate and normalization
    conj!(workspace.x_work)
    fft!(y, workspace.x_work, use_ivdep)
    conj!(y)
    y ./= n
    
    return y
end

end