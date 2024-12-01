module CatabraFFT

include("kron.jl")

__precompile__()
GC.gc()

export FFT!, F_cache

# Non-mutating wrapper that reuses preallocated workspace
struct FFTWorkspace
    x_work::Vector{ComplexF64}
    function FFTWorkspace(n::Int)
        new(Vector{ComplexF64}(undef, n))
    end
end

# Thread-local workspace to avoid allocations in parallel code
const WORKSPACE = Dict{Int, FFTWorkspace}()

# Get or create workspace for a given size
@inline function get_workspace(n::Int)
    get!(WORKSPACE, n) do
        FFTWorkspace(n)
    end
end

# Non-mutating FFT that reuses workspace
@inline function FFT(x::AbstractVector{ComplexF64})
    n = length(x)
    workspace = get_workspace(n)
    copyto!(workspace.x_work, x)  # Fast copy into preallocated space
    y = similar(x)
    FFT!(y, workspace.x_work)
    return y
end
end
