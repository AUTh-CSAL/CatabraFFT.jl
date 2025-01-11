include("radix_factory.jl")
using RuntimeGeneratedFunctions
using LoopVectorization, Primes, BenchmarkTools

RuntimeGeneratedFunctions.init(@__MODULE__)

# Module to store all runtime-generated FFT functions
module FFTStorage
    using RuntimeGeneratedFunctions
    using ..RadixFFTCompositor

    RuntimeGeneratedFunctions.init(@__MODULE__)
    
    # Cache for compiled FFT functions
    const FFT_CACHE = Dict{Int, Function}()
    
    # Function to get or create FFT function for size n
    function get_or_create_fft(n::Int, T::Type{<:AbstractFloat}=Float64)
        get!(FFT_CACHE, n) do
            module_code = RadixFFTCompositor.create_kernel_module(n, T)
            module_expr = Meta.parse(module_code)
            
            # Evaluate the module in the current context
            Core.eval(@__MODULE__, module_expr)
            
            # Create wrapper function using RuntimeGeneratedFunctions
            func_name = Symbol("fft$(n)_shell!")
            wrapper_code = """
            function (Y::Vector{Complex{$T}}, X::Vector{Complex{$T}})
                radix_2_family.$func_name(Y, X)
            end
            """
            
            # Compile and return the function
            expr = Meta.parse(wrapper_code)
            @RuntimeGeneratedFunction(expr)
        end
    end
end

using LoopVectorization, Primes, BenchmarkTools


#RuntimeGeneratedFunction.init(@_MODULE)

include("radix_factory.jl")
# Cache-oblivious matrix transpose implementation
function transpose_recursive!(A::AbstractMatrix, B::AbstractMatrix, 
                           start_row_A=1, start_col_A=1,
                           start_row_B=1, start_col_B=1,
                           n_rows=size(A,1), n_cols=size(A,2))
    
    if n_rows <= 8 || n_cols <= 8  # Base case threshold
        for i in 0:n_rows-1
            for j in 0:n_cols-1
                B[start_row_B+j, start_col_B+i] = A[start_row_A+i, start_col_A+j]
            end
        end
        return
    end
    
    # Recursive divide-and-conquer
    if n_rows >= n_cols
        mid = n_rows ÷ 2
        transpose_recursive!(A, B, 
                           start_row_A, start_col_A,
                           start_row_B, start_col_B,
                           mid, n_cols)
        transpose_recursive!(A, B,
                           start_row_A + mid, start_col_A,
                           start_row_B, start_col_B + mid,
                           n_rows - mid, n_cols)
    else
        mid = n_cols ÷ 2
        transpose_recursive!(A, B,
                           start_row_A, start_col_A,
                           start_row_B, start_col_B,
                           n_rows, mid)
        transpose_recursive!(A, B,
                           start_row_A, start_col_A + mid,
                           start_row_B + mid, start_col_B,
                           n_rows, n_cols - mid)
    end
end

# In-place matrix transpose using the recursive algorithm
function transpose_inplace!(A::AbstractMatrix)
    n_rows, n_cols = size(A)
    B = similar(A)
    transpose_recursive!(A, B)
    copyto!(A, B)
end

function find_closest_factors(n::Int)
    if isprime(n)
        return 1, n
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

# Helper function to get the appropriate radix-2 FFT function
function get_radix_fft_function(n::Int, radix_2_family::Module)
    function_name = Symbol("fft$(n)_shell!")
    if isdefined(radix_2_family, function_name)
        return getfield(radix_2_family, function_name)
    end
    return nothing
end

# Cache-oblivious FFT implementation with radix-2 support
function fft_cache_oblivious!(X::Vector{Complex{Float64}})
    n = length(X)
    
    # Check if we have a specialized radix-2 implementation for this size
    #=
    radix_fft = get_radix_fft_function(n)
    if !isnothing(radix_fft)
        Y = similar(X)
        radix_fft(Y, X)
        copyto!(X, Y)
        return X
    end
    =#
    
    # If no specialized implementation, use the cache-oblivious approach
    if n <= 8  # Base case
        return fft_direct!(X)
    end
    
    # Factor n into n1 * n2 where n1 ≈ n2 ≈ √n
    n1, n2 = find_closest_factors(n)
    
    # Step 1: View as matrix and transpose
    A = reshape(X, n1, n2)
    transpose_inplace!(A)
    
    # Step 2: Compute n2 FFTs of size n1
    for i in 1:n2
        fft_cache_oblivious!(A[:,i])
    end
    
    # Step 3: Apply twiddle factors
    for i in 0:n1-1
        for j in 0:n2-1
            ω = exp(-2π * im * (i*j) / n)
            A[i+1,j+1] *= ω
        end
    end
    
    # Step 4: Transpose again
    transpose_inplace!(A)
    
    # Step 5: Compute n1 FFTs of size n2
    for i in 1:n1
        fft_cache_oblivious!(A[:,i])
    end
    
    # Step 6: Final transpose
    transpose_inplace!(A)
    
    return X
end

# Direct FFT computation for small inputs
function fft_direct!(X::Vector{Complex{Float64}})
    n = length(X)
    Y = similar(X)
    fft_func = FFTStorage.get_or_create_fft(n, Float64)
    Base.invokelatest(fft_func, Y, X)
end

function benchmark_fft(n::Int)
    X = rand(Complex{Float64}, n)
    @benchmark fft_cache_oblivious!($X)
end

# Example usage
function example_usage()
    # Test with power of 2 size
    n = 2^8
    X = rand(Complex{Float64}, n)
    
    # First call will compile the function
    result1 = fft_cache_oblivious!(copy(X))
    
    # Subsequent calls use cached function
    result2 = fft_cache_oblivious!(copy(X))
    
    # Benchmark
    b = benchmark_fft(n)
    println("Median time: ", median(b).time, " ns")
    return b
end

example_usage()