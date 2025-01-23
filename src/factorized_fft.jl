include("radix_factory.jl")
include("matrix_operations.jl")


# Module to store all runtime-generated FFT functions
module FFTWorkspace
    using RuntimeGeneratedFunctions, Primes
    using ..RadixGenerator, ..MatrixOperations

    RuntimeGeneratedFunctions.init(@__MODULE__)
    
    # Cache for compiled FFT functions
    const FFT_CACHE = Dict{Tuple{Int, Bool}, Function}()

    function find_closest_factors(n::Int)
        isprime(n) && return 1, n
    
        p = isqrt(n) # Start with p as the floor of sqrt(n)
        while n % p != 0 # Adjust p until it divides n evenly
            p -= 1
            p == 1 && throw(ArgumentError("Unable to find non-prime factors for $n"))
        end
        return p, div(n, p)
    end
    
    @inline function D(p,m, ::Type{T})::Matrix where {T<:AbstractFloat}
        w = cispi.(T(-2/(p*m)) * collect(1:m-1))
        d = zeros(Complex{T},(p-1)*(m-1))
    
        @inbounds d[1:m-1] .= w
    
        @inbounds @simd for j in 2:p-1
              @views d[(j-1)*(m-1)+1:j*(m-1)] .= w .* view(d, (j-2)*(m-1)+1:(j-1)*(m-1))
        end
    
        return reshape(d, m-1, p-1)
    end

    # Function to get or create FFT function for size n
    function get_or_create_fft(n::Int, ::Type{T}, eo::Bool) where T <: AbstractFloat
        key = (n, eo)
        haskey(FFT_CACHE, key) && return FFT_CACHE[key]
        get!(FFT_CACHE, key) do

            # Must have already loaded generated module RadixGenerator
            # Create wrapper function using RuntimeGeneratedFunctions
            func_name = eo ? Symbol("fft$(n)_shell!") : Symbol("fft$(n)_shell_y!")
            wrapper_code = """
                function $func_name(Y::AbstractVector{Complex{$T}}, X::AbstractVector{Complex{$T}})
                    radix_2_family.$func_name(Y, X)
                end
            """
            @show n, func_name
            # Compile and return the function
            eval(Meta.parse(wrapper_code))
        end
    end

    function fft_kernel!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, n::Int, eo::Bool) where T <: AbstractFloat 
        fft_func = get_or_create_fft(n, T, eo)
        println("fft kernel:")
        @show fft_func, n
        @show x, y
        #fft_func(y, x) # Codelet execution
        Base.invokelatest(fft_func, y, x) # Codelet execution
    end

    @inline function fft_cache_oblivious!(Y::AbstractArray{Complex{T}, 1}, X::AbstractArray{Complex{T}, 1}, n::Int, eo::Bool) where T <: AbstractFloat
    
    n <= 4 && return fft_kernel!(Y, X, n, eo) 
        
    # Factor n into n1 * n2 where n1 ≈ n2 ≈ √n
    n1, n2 = find_closest_factors(n)
    println("n1 = $n1, n2 = $n2")
    
    # Step 0: View as matrix and transpose
    A = reshape(X, n1, n2)
    B = reshape(Y, n1, n2)
    
    # Step 1: Compute n2 FFTs of size n1
    @inbounds @simd for i in 1:n2
        fft_cache_oblivious!(B[:, i], A[:,i], n1, !eo)
    end

    
    d = D(n1, n2, T)
    # Step 2: Apply twiddle factors
    @inbounds @simd for i in 1:n1
        @inbounds @simd for j in 1:n2
            B[i,j] *= d[(i-1)*(n2-1) + j]
        end
    end
    
    # Step 3: Transpose again
    #MatrixOperations.transpose!(A, n2, n1)
    #MatrixOperations.transpose!(Y, n2, n1)
    #@views A.= transpose(B)
    
    # Step 4: Compute n1 FFTs of size n2
    @inbounds @simd for i in 1:n1
        fft_cache_oblivious!(Y[:,i], A[:,i], n2, !eo)
    end
    
    # Step 5: Final transpose
    #MatrixOperations.transpose!(A, n1, n2)
    #MatrixOperations.transpose!(Y, n1, n2)
    @views B .= transpose(A)
    #return X
end
end

module Testing
include("radix_factory.jl")
using FFTW, BenchmarkTools

using ..FFTWorkspace
@show isdefined(Main, :FFTWorkspace)

n = 2^3
CType = Float64
x= [Complex{CType}(i,i) for i in 1:n];
y = similar(x);
RadixGenerator.evaluate_fft_generated_module(Testing, n, CType)
FFTWorkspace.fft_cache_oblivious!(y, x, n, false)

F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)
yy = F * x

@show y, yy
@assert y ≈ yy
factory_b = @benchmark FFTWorkspace.fft_cache_oblivious!($y, $x, $n, false)
fftw_b = @benchmark $F * $x

println("Custom FFT function benchmark:")
display(factory_b)  # Use `display` for detailed results
println("\nFFTW benchmark:")
display(fftw_b)

end