module CatabraFFT

include("kron.jl")

__precompile__()
GC.gc()

#export fft!

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

@doc """
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
    fft!(y, workspace.x_work)
    return y
end


@doc """
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
    fft!(y, workspace.x_work)
    conj!(y)
    y ./= n
    
    return y
end

end
