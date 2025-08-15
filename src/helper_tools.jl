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

function return_sorted_prime_powers(n::Int)
    primes = [2,3,5,7] # Primes I have families of
    prime_powers = []

    for prime in primes
        pow = prime
        while pow <= n
            push!(prime_powers, pow)
            if pow > typemax(Int) ÷ prime #stack overflow protection
                break
            end
            pow *= prime
        end
    end

    # Insertion sort (descending order)
    sort!(prime_powers, rev=true)
    return prime_powers
end

#When p ≈ m fewer matrices are recomputed => better runtime.
# Special strided FFT kernels with lower radix rank for special computation of n
# => mixed-radix-(m,p) !!!
function find_closest_factors(n::Int, prime_powers_preference=true)
    if isprime(n)
        return 1, n
    end
    if prime_powers_preference
        prime_powers = return_sorted_prime_powers(n)

        for p in prime_powers
            if n % p == 0
                return p, div(n,p)
            end
        end
    end
    p = isqrt(n) # Start with p as the floor of sqrt(n)

    while n % p != 0 # Adjust p until it divides n evenly
        p -= 1
        if p == 1
            error("Unable to find non-prime factors for $n")
        end
    end
    return p, div(n, p)
end