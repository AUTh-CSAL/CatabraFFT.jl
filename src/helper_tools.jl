#Helper Tools for CatabraFFT
using SIMD

# Custom view type for zero-allocation reshaping of vectors to matrices
struct StaticReshapedArray{T,N,AA<:AbstractArray} <: AbstractArray{T,N}
    parent::AA
    dims::NTuple{N,Int}
    
    function StaticReshapedArray{T,N,AA}(parent::AA, dims::NTuple{N,Int}) where {T,N,AA<:AbstractArray}
        prod(dims) == length(parent) || throw(DimensionMismatch("New dimensions $(dims) must be consistent with array length $(length(parent))"))
        new{T,N,AA}(parent, dims)
    end
end

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

@inline function is_power_of(n::Int, p::Int)
    @inbounds while n > 1
        n % p != 0 && return false
        n ÷= p
    end
    return true
end

function subpowers_of_two(N::Int)
    @assert N > 1 && (N & (N - 1)) == 0 "N must be a power of two greater than 1"
    
    # Pre-allocate with known size
    log2N = trailing_zeros(N)
    subpowers = Vector{Int}(undef, log2N)
    
    @inbounds for i in 1:log2N
        subpowers[i] = N >> (i - 1)
    end
    
    return subpowers
end

function get_radix_family(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    if ispow2(radix)
        return radix_2_family
    elseif radix ∈ (3, 9)
        return radix_3_family
    elseif radix == 5
        return radix_5_family
    elseif radix == 7
        return radix_7_family
    else
        error("Unsupported radix: $radix")
    end
end

@inline function get_radix_divisor(op_type::Symbol)
    return parse(Int, String(op_type)[4:end])
end

function get_function_reference(radix_family, base_function_name::Symbol)
    isdefined(radix_family, base_function_name) || 
        error("Function $base_function_name not found in module $(radix_family)")
    return getfield(radix_family, base_function_name)
end

function return_sorted_prime_powers(n::Int)
    primes = (2, 3, 5, 7)  # Use tuple for immutability
    prime_powers = Int[]
    
    @inbounds for prime in primes
        pow = prime
        while pow <= n
            push!(prime_powers, pow)
            pow > typemax(Int) ÷ prime && break
            pow *= prime
        end
    end
    
    sort!(prime_powers, rev=true)
    return prime_powers
end

function find_closest_factors(n::Int, prime_powers_preference::Bool=true)
    isprime(n) && return (1, n)
    
    if prime_powers_preference
        prime_powers = return_sorted_prime_powers(n)
        
        @inbounds for p in prime_powers
            n % p == 0 && return (p, n ÷ p)
        end
    end
    
    # Fallback to square root method
    p = isqrt(n)
    @inbounds while p > 1
        n % p == 0 && return (p, n ÷ p)
        p -= 1
    end
    
    error("Unable to find non-prime factors for $n")
end

@inline function signflip(v::Vec{N,T}, mask::Vec{N,UInt32}) where {N, T<:AbstractFloat}
    # Reinterpret float as uint, XOR with sign bit, reinterpret back.
    # Julia doesn't like bitwise operations on its floats...
    v_uint = reinterpret(Vec{N,UInt32}, v)
    v_flipped = v_uint ⊻ mask
    return reinterpret(Vec{N,T}, v_flipped)
end