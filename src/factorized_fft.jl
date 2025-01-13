include("radix_factory.jl")

module MatrixOperations

using LoopVectorization, ..RadixGenerator

# Vectorized in-place transpose for small matrices
function transpose_small!(a::AbstractArray{Complex{T}, 2}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:i-1
            tmp = a[i,j]
            a[i,j] = a[j,i]
            a[j,i] = tmp
        end
    end
end

# Cache-oblivious transpose with vectorized base case
function transpose!(a::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 8
        transpose_small!(a, n)
    else
        k = div(n, 2)
        transpose!(view(a, 1:k, 1:k), k, N)
        transpose!(view(a, 1:k, k+1:n), k, N)
        transpose!(view(a, k+1:n, 1:k), k, N)
        transpose!(view(a, k+1:n, k+1:n), k, N)
        
        @inbounds @simd for i in 1:k
            @inbounds @simd for j in 1:k
                tmp = a[i, j+k]
                a[i, j+k] = a[i+k, j]
                a[i+k, j] = tmp
            end
        end
        
        if isodd(n)
            @inbounds @simd for i in 1:n-1
                tmp = a[i,n]
                a[i,n] = a[n,i]
                a[n,i] = tmp
            end
        end
    end
end

# Vectorized base case for standard matrix multiplication
@inline function matmul_small!(a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, c::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            cij = zero(T)
            @inbounds @simd for k in 1:n
                cij += a[i,k] * b[k,j]
            end
            c[i,j] += cij
        end
    end
end

# Cache-oblivious standard matrix multiplication with vectorized base case
function matmul!(a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, c::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 16
        matmul_small!(a, b, c, n)
    else
        k = div(n, 2)
        
        matmul!(view(a, 1:k, 1:k), view(b, 1:k, 1:k), view(c, 1:k, 1:k), k, N)
        matmul!(view(a, 1:k, k+1:n), view(b, k+1:n, 1:k), view(c, 1:k, 1:k), k, N)
        
        matmul!(view(a, 1:k, 1:k), view(b, 1:k, k+1:n), view(c, 1:k, k+1:n), k, N)
        matmul!(view(a, 1:k, k+1:n), view(b, k+1:n, k+1:n), view(c, 1:k, k+1:n), k, N)
        
        matmul!(view(a, k+1:n, 1:k), view(b, 1:k, 1:k), view(c, k+1:n, 1:k), k, N)
        matmul!(view(a, k+1:n, k+1:n), view(b, k+1:n, 1:k), view(c, k+1:n, 1:k), k, N)
        
        matmul!(view(a, k+1:n, 1:k), view(b, 1:k, k+1:n), view(c, k+1:n, k+1:n), k, N)
        matmul!(view(a, k+1:n, k+1:n), view(b, k+1:n, k+1:n), view(c, k+1:n, k+1:n), k, N)
        
        if isodd(n)
            @inbounds @simd for i in 1:n
                @inbounds @simd for j in 1:n
                    @inbounds @simd for k in ((i <= n-1 && j <= n-1) ? n : 1):n
                        c[i,j] += a[i,k] * b[k,j]
                    end
                end
            end
        end
    end
end

# Strassen algorithm implementation
@inline function strassen_add!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] + b[i,j]
        end
    end
end

@inline function strassen_sub!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] - b[i,j]
        end
    end
end

function strassen_mul!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 16  # Use standard multiplication for small matrices
        matmul_small!(a, b, c, n)
    else
        k = div(n, 2)
        
        # Temporary matrices for Strassen algorithm
        m1 = zeros(T, k, k)
        m2 = zeros(T, k, k)
        m3 = zeros(T, k, k)
        m4 = zeros(T, k, k)
        m5 = zeros(T, k, k)
        m6 = zeros(T, k, k)
        m7 = zeros(T, k, k)
        
        temp1 = zeros(T, k, k)
        temp2 = zeros(T, k, k)
        
        # M1 = (A11 + A22)(B11 + B22)
        strassen_add!(temp1, view(a, 1:k, 1:k), view(a, k+1:n, k+1:n), k)
        strassen_add!(temp2, view(b, 1:k, 1:k), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m1, temp1, temp2, k, N)
        
        # M2 = (A21 + A22)B11
        strassen_add!(temp1, view(a, k+1:n, 1:k), view(a, k+1:n, k+1:n), k)
        strassen_mul!(m2, temp1, view(b, 1:k, 1:k), k, N)
        
        # M3 = A11(B12 - B22)
        strassen_sub!(temp1, view(b, 1:k, k+1:n), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m3, view(a, 1:k, 1:k), temp1, k, N)
        
        # M4 = A22(B21 - B11)
        strassen_sub!(temp1, view(b, k+1:n, 1:k), view(b, 1:k, 1:k), k)
        strassen_mul!(m4, view(a, k+1:n, k+1:n), temp1, k, N)
        
        # M5 = (A11 + A12)B22
        strassen_add!(temp1, view(a, 1:k, 1:k), view(a, 1:k, k+1:n), k)
        strassen_mul!(m5, temp1, view(b, k+1:n, k+1:n), k, N)
        
        # M6 = (A21 - A11)(B11 + B12)
        strassen_sub!(temp1, view(a, k+1:n, 1:k), view(a, 1:k, 1:k), k)
        strassen_add!(temp2, view(b, 1:k, 1:k), view(b, 1:k, k+1:n), k)
        strassen_mul!(m6, temp1, temp2, k, N)
        
        # M7 = (A12 - A22)(B21 + B22)
        strassen_sub!(temp1, view(a, 1:k, k+1:n), view(a, k+1:n, k+1:n), k)
        strassen_add!(temp2, view(b, k+1:n, 1:k), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m7, temp1, temp2, k, N)
        
        # C11 = M1 + M4 - M5 + M7
        strassen_add!(view(c, 1:k, 1:k), m1, m4, k)
        strassen_sub!(view(c, 1:k, 1:k), view(c, 1:k, 1:k), m5, k)
        strassen_add!(view(c, 1:k, 1:k), view(c, 1:k, 1:k), m7, k)
        
        # C12 = M3 + M5
        strassen_add!(view(c, 1:k, k+1:n), m3, m5, k)
        
        # C21 = M2 + M4
        strassen_add!(view(c, k+1:n, 1:k), m2, m4, k)
        
        # C22 = M1 - M2 + M3 + M6
        strassen_add!(view(c, k+1:n, k+1:n), m1, m6, k)
        strassen_sub!(view(c, k+1:n, k+1:n), view(c, k+1:n, k+1:n), m2, k)
        strassen_add!(view(c, k+1:n, k+1:n), view(c, k+1:n, k+1:n), m3, k)
    end
end
end

# Module to store all runtime-generated FFT functions
module FFTWorkspace
    using RuntimeGeneratedFunctions, Primes
    using ..RadixGenerator, ..MatrixOperations

    RuntimeGeneratedFunctions.init(@__MODULE__)
    
    # Cache for compiled FFT functions
    const FFT_CACHE = Dict{Int, Function}()

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

    function evaluate_fft_generated_module(n::Int, ::Type{T}) where T <: AbstractFloat
        module_expr = RadixGenerator.parse_module(RadixGenerator.create_kernel_module(n, T))
        @show module_expr
        
        # Evaluate the module in the current context
        Core.eval(@__MODULE__, module_expr)
    end

    # Function to get or create FFT function for size n
    function get_or_create_fft(n::Int, T::Type{<:AbstractFloat}=Float63)
        haskey(FFT_CACHE, n) && return FFT_CACHE[n]
        get!(FFT_CACHE, n) do

            # Must have already loaded generated module RadixGenerator
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

    function fft_kernel!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int) where T <: AbstractFloat 
        fft_func = get_or_create_fft(n, T)
        fft_func(y, x) # Codelet execution
    end

    @inline function fft_cache_oblivious!(Y::AbstractArray{Complex{T}, 1}, X::AbstractArray{Complex{T}, 1}) where T <: AbstractFloat
    n = length(X)
    @show n
    
    # If no specialized implementation, use the cache-oblivious approach
    if n <= 4  # Base case
        return fft_kernel!(Y, X, n)
    end
    
    # Factor n into n0 * n2 where n1 ≈ n2 ≈ √n
    n1, n2 = find_closest_factors(n)
    println("n1 = $n1, n2 = $n2")
    
    # Step 0: View as matrix and transpose
    Y, A = reshape(Y, n1, n2), reshape(X, n1, n2)
    MatrixOperations.transpose!(A, n1, n2)
    MatrixOperations.transpose!(Y, n1, n2)
    
    # Step 1: Compute n2 FFTs of size n1
    @inbounds @simd for i in 1:n2
        fft_cache_oblivious!(Y[:, i], A[:,i])
    end

    #D = D(n1, n2, T)
    p, m = n1, n2
    w = cispi.(T(-2/(p*m)) * collect(1:m-1))
    d = zeros(Complex{T},(p-1)*(m-1))

    @inbounds d[1:m-1] .= w

    @inbounds @simd for j in 2:p-1
          @views d[(j-1)*(m-1)+1:j*(m-1)] .= w .* view(d, (j-2)*(m-1)+1:(j-1)*(m-1))
    end
    
    # Step 2: Apply twiddle factors
    @inbounds @simd for i in 0:n1-1
        @inbounds @simd for j in 0:n2-1
            A[i+1,j+1] *= d[n1*i + j]
        end
    end
    
    # Step 3: Transpose again
    MatrixOperations.transpose!(A, n1, n2)
    MatrixOperations.transpose!(Y, n1, n2)
    
    # Step 4: Compute n1 FFTs of size n2
    @inbounds @simd for i in 1:n1
        fft_cache_oblivious!(Y[:,i], A[:,i])
    end
    
    # Step 5: Final transpose
    MatrixOperations.transpose!(A, n1, n2)
    MatrixOperations.transpose!(Y, n1, n2)
    #return X
end


end

using FFTW, BenchmarkTools
#using .FFTWorkspace

using .FFTWorkspace
@show isdefined(Main, :FFTWorkspace)

n = 2^4
ctype = Float64
x= [Complex{ctype}(i,i) for i in 1:n];
y=  similar(x);
FFTWorkspace.evaluate_fft_generated_module(n, ctype)
FFTWorkspace.fft_cache_oblivious!(y, x)

F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)
yy = F * x

@show y, yy
@assert y ≈ yy
factory_b = @benchmark FFTWorkspace.fft_cache_oblivious!($y, $x)
fftw_b = @benchmark $F * $x

println("Custom FFT function benchmark:")
display(factory_b)  # Use `display` for detailed results

println("\nFFTW benchmark:")
display(fftw_b)
