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
const WORKSPACE = Dict{Int, FFTWorkspace}()

# Get or create workspace for a given size
@inline function get_workspace(n::Int, ::Type{T})::FFTWorkspace where {T<:AbstractFloat}
    get!(WORKSPACE, n) do
        FFTWorkspace(n, T)
    end
end

# Non-mutating FFT that reuses workspace
@inline function fft(x::AbstractVector{Complex{T}})::AbstractVector{Complex{T}} where {T<:AbstractFloat}
    n = length(x)
    workspace = get_workspace(n, T)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    fft!(y, workspace.x_work)
    return y
end
end
