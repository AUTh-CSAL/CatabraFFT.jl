# Helper Tools used throught this project:

# Custom view type for zero-allocation reshaping of vectors to matrices
struct StaticReshapedArray{T,N,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::AA
    dims::NTuple{N,Int}
    
    # Inner constructor to verify dimensions
    function StaticReshapedArray{T,N,AA}(parent::AA, dims::NTuple{N,Int}) where {T,N,AA<:AbstractArray}
        prod(dims) == length(parent) || throw(DimensionMismatch("New dimensions $(dims) must be consistent with array length $(length(parent))"))
        new{T,N,AA}(parent, dims)
    end
end

# Outer constructor
function static_reshape(arr::AbstractArray{T}, dims::Vararg{Int,N}) where {T,N}
    StaticReshapedArray{T,N,typeof(arr)}(arr, dims)
end

# Implement required Array interface
Base.size(A::StaticReshapedArray) = A.dims
Base.parent(A::StaticReshapedArray) = A.parent

@inline function Base.getindex(A::StaticReshapedArray{T,2}, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    @inbounds A.parent[(j-1)*size(A,1) + i]
end

@inline function Base.setindex!(A::StaticReshapedArray{T,2}, v, i::Int, j::Int) where T
    @boundscheck checkbounds(A, i, j)
    @inbounds A.parent[(j-1)*size(A,1) + i] = v
    v
end

function is_power_of(n::Int, p::Int)
    while n > 1
        if n % p != 0
            return false
        end
        n ÷= p
    end
    return true
end

function subpowers_of_two(N::Int)
    # Check if N is a power of two
    @assert N > 1 && (N & (N - 1)) == 0 "N must be a power of two greater than 1"
    
    # Generate the list of subpowers
    subpowers = Vector{Int}()
    while N >= 2
        push!(subpowers, N)
        N = div(N, 2)
    end
    return subpowers
end

function get_radix_family(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    if ispow2(radix)
        return radix_2_family
    elseif radix ∈ [3, 9]
        return radix_3_family
    elseif radix == 5
        return radix_5_family
    elseif radix == 7
        return radix_7_family
    else
        error("Unsupported radix: $radix")
    end
end

# Int mapping for specific symbol naming
function get_radix_divisor(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    return radix
end


function get_function_reference(radix_family, base_function_name::Symbol)
    func = getfield(radix_family, base_function_name)
    if !isdefined(radix_family, base_function_name)
        error("Function $base_function_name not found in module $(radix_family)")
    end
    return func
end
