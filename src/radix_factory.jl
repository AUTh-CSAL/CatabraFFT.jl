module RadixFFTCompositor

using LoopVectorization

function generate_module_constants(n::Int, ::Type{T}) where T <: AbstractFloat
    @assert ispow2(n) "n must be a power of 2"
    str = "# Optimized twiddle factors for radix-2^s FFT size $n\n\n"
    current_n = n
    # Only store the minimal set of unique twiddle factors needed
    while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        # Store only unique twiddle factors for this stage
        # We exploit symmetry and periodicity to minimize storage
        str *= "# Stage $current_n constants\n"
        for i in 1:2:s
            # Calculate angle once and reuse
            angle = 2 * (n4-i) / current_n
            angle_cis_1 = Complex{T}(cispi(angle))
            angle_cis_2 = Complex{T}(cispi(-angle))
            str *= "const CISPI_$(n4-i)_$(n2)_Q1::Complex{$T} = $angle_cis_1\n"
            str *= "const CISPI_$(n4-i)_$(n2)_Q4::Complex{$T} = $angle_cis_2\n"
        end
        str *= "\n"
        current_n >>= 1
    end
    
    # Add only essential special constants
    if n >= 8
        str *= "const INV_SQRT2_Q1 = $(Complex{T}(1/sqrt(2) + im * 1/sqrt(2)))\n"
        str *= "const INV_SQRT2_Q4 = $(Complex{T}(1/sqrt(2) - im * 1/sqrt(2)))\n"
        #str *= "const INV_SQRT2_Q4 = $(T(1/sqrt(2)*(1 - im)))\n"
    end
    
    return str
end

"""
Enhanced twiddle factor expression generator with improved constant recognition
"""
function get_constant_expression(w::Complex{T}, n::Integer)::String where T <: AbstractFloat
    real_part = real(w)
    imag_part = imag(w)
    
    # Helper for approximate equality
    isclose(a, b) = (abs(real(a) - real(b)) < eps(T) * 10) && (abs(imag(a) - imag(b)) < eps(T) * 10)
    
    # Function to get sign string
    sign_str(x) = x ≥ 0 ? "+" : "-"
    
    # Common cases table with twiddle factors patterns commonly met

    common_cases = [
        (1.0, 0.0) => "1",
        (-1.0, 0.0) => "-1",
        (0.0, 1.0) => "im",
        (0.0, -1.0) => "-im",
        (1/√2, 1/√2) => "INV_SQRT2_Q1",
        (1/√2, -1/√2) => "INV_SQRT2_Q4",
        (-1/√2, 1/√2) => "-INV_SQRT2_Q4",
        (-1/√2, -1/√2) => "-INV_SQRT2_Q1"
    ]
    
    # Check special cases first
    for ((re, im), expr) in common_cases
        if isclose(real_part, re) && isclose(imag_part, im)
            return expr
        end
    end
    
    current_n = n
    # Handle cases based on radix size
    while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        angles = [(n4-i,n2) for i in 1:2:s]
        for (num, den) in angles
            #cp = cospi(num/den)
            #sp = sinpi(num/den)
            cispi1, cispi2  = cispi(num/den), cispi(-num/den)
            if isclose(w, cispi1)
                return "CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, -cispi1)
                return "-CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, cispi2)
                return "CISPI_$(num)_$(den)_Q4"
            elseif isclose(w, -cispi2)
                return "-CISPI_$(num)_$(den)_Q4"
            elseif isclose(w, -im*cispi1)
                return "-im*CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, -im*cispi2)
                return "-im*CISPI_$(num)_$(den)_Q4"
            end
        end
        current_n >>= 1
    end
    
    # Fallback to numerical representation with high precision
    return "($(round(real_part, digits=16))$(sign_str(imag_part))$(abs(round(imag_part, digits=16)))*im)"
end

"""
Generate twiddle factor expressions for a given collection of indices
"""
function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

# Courtesy of Nikos Pitsianis for 'recfft2'
############################################

function inccounter()
  let counter = 0
    return () -> begin
      counter += 1
      return counter
    end
  end
end

inc = inccounter()


function recfft2(y, x, w=nothing)
  n = length(x)
  # println("n = $n, x = $x, y = $y")
  if n == 1
    ""
  elseif n == 2
    if isnothing(w)
      s = """
         $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(x[1]) - $(x[2])
         """
    else
      s = """
         $(y[1]), $(y[2]) = ($(w[1]))*($(x[1]) + $(x[2])), ($(w[2]))*($(x[1]) - $(x[2]))
         """
    end

    return s
  else
    t = vmap(i -> "t$(inc())", 1:n)
    n2 = n ÷ 2
    wn = get_twiddle_expression(collect(0:n2-1), n)

    s1 = recfft2(t[1:n2], x[1:2:n])
    s2 = recfft2(t[n2+1:n], x[2:2:n], wn)

    if isnothing(w)
      s3p = foldl(*, vmap(i -> ",$(y[i])", 2:n2); init="$(y[1])") *
            " = " *
            foldl(*, vmap(i -> ",$(t[i]) + $(t[i+n2])", 2:n2), init="$(t[1]) + $(t[1+n2])") * "\n"
      s3m = foldl(*, vmap(i -> ",$(y[i+n2])", 2:n2); init="$(y[n2+1])") *
            " = " *
            foldl(*, vmap(i -> ",$(t[i]) - $(t[i+n2])", 2:n2), init="$(t[1]) - $(t[1+n2])") * "\n"
    else
      s3p = foldl(*, vmap(i -> ", $(y[i])", 2:n2); init="$(y[1])") *
            " = " *
            foldl(*, vmap(i -> ", ($(w[i]))*($(t[i]) + $(t[i+n2]))", 2:n2), init="($(w[1]))*($(t[1]) + $(t[1+n2]))") * "\n"
      s3m = foldl(*, vmap(i -> ", $(y[i+n2])", 2:n2); init="$(y[n2+1])") *
            " = " *
            foldl(*, vmap(i -> ", ($(w[n2+i]))*($(t[i]) - $(t[i+n2]))", 2:n2), init="($(w[n2+1]))*($(t[1]) - $(t[1+n2]))") * "\n"
    end

    return s1 * s2 * s3p * s3m
  end
end

function makefftradix(n, ::Type{T}) where T <: AbstractFloat
  x = vmap(i -> "#INPUT#[$i]", 1:n)
  y = vmap(i -> "#OUTPUT#[$i]", 1:n)
  s = recfft2(y, x)
  return s
end

# Function to generate kernel name
function generate_kernel_name(radix::Int, suffixes::Vector{String})
    base = "fft$(radix)_shell"
    suffix = join(suffixes, "_")
    return string(base, !isempty(suffix) ? "_$suffix" : "", "!")
end

# Function to generate function signature
function generate_signature(suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    y_only = "y" in suffixes
    layered = "layered" in suffixes
    
    if y_only
        return "(y::AbstractVector{Complex{$T}}, s::Int)"
    elseif layered
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int, n1::Int, theta::$T=0.125)"
    else
        #return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int)"
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})"
    end
end

# Function to generate loop decorators
function generate_loop_decorators(suffixes::Vector{String})
    decorators = ["@inbounds"]
    if "ivdep" in suffixes
        push!(decorators, "@simd ivdep")
    else
        push!(decorators, "@simd")
    end
    return join(decorators, " ")
end

# Main function to generate kernel code
function generate_kernel(radix::Int, suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    name = generate_kernel_name(radix, suffixes)
    signature = generate_signature(suffixes, T)
    decorators = generate_loop_decorators(suffixes)
    
    kernel_pattern = makefftradix(radix, T)
    input = "y" in suffixes ? "y" : "x"
    output = "y"
    
    # Replace placeholders in kernel pattern
    kernel_code = replace(kernel_pattern, 
        "#INPUT#" => input,
        "#OUTPUT#" => output)
    
    # Generate the complete function
    if "layered" in suffixes
        # Special handling for layered kernels
        return generate_layered_kernel(name, signature, decorators, kernel_code, radix)
    else
        return """
        @inline function $name$signature
            @inbounds $kernel_code
        end
        """
    end
end

# Helper function to generate layered kernel
function generate_layered_kernel(name, signature, decorators, kernel_code, radix)
    s = log2(radix)  # Assuming radix is a power of 2
    @assert isinteger(s) && s > 0 "Radix must be a power of 2"
    
    # Generate twiddle factor computation dynamically for any radix
    twiddle_code = String[]
    for i in 2:radix-1
        twiddle_expression = "w$(div(i, 2))p * w$(div(i + 1, 2))p"
        push!(twiddle_code, "w$(i)p = $twiddle_expression")
    end
    twiddle_code_str = join(twiddle_code, "\n")
    
    return """
    @inline function $name$signature

        #TODO ADD INIT KERNEL

        # Section with twiddle factors
        $decorators for p in 1:(n1-1)
            $twiddle_code_str
            
            $decorators for q in 1:s
                $kernel_code
            end
        end
    end
    """
end

# Function to generate all possible kernel combinations
function generate_all_kernels(N::Int, ::Type{T}) where T <: AbstractFloat
    if N < 2 || (N & (N - 1)) != 0  # Check if N is less than 2 or not a power of 2
        error("N must be a power of 2 and greater than or equal to 2")
    end
        
    radices = []
    current = 2
    while current <= N
        push!(radices, current)
        current *= 2
    end

    suffix_combinations = [
        String[],
        #["ivdep"],
        #["y"],
        #["y", "ivdep"],
        #["layered"],
        #["layered", "ivdep"]
    ]
    
    #kernels = Dict{String, String}()
    kernels = Vector{String}()
    
    for radix in radices
        for suffixes in suffix_combinations
            #name = generate_kernel_name(radix, suffixes)
            code = generate_kernel(radix, suffixes, T)
            #kernels[name] = code
            push!(kernels, code)
        end
    end
    
    #@show kernels
    
    return kernels
end

# Function to evaluate and create the functions in a module
function create_kernel_module(N::Int, ::Type{T}) where T <: AbstractFloat
    module_constants = generate_module_constants(N, T)
    kernels = generate_all_kernels(N, T)
    
    family_module_code = """
    module radix_2_family
        using LoopVectorization
        
        $module_constants
        
        $(join(kernels, "\n\n"))
    end
    """
    
    return family_module_code
end

end

#txt = RadixFFTCompositor.create_kernel_module(32, Float64)
#code = Meta.parse(txt)
#@show code
#radix_2_family = eval(code)

using RuntimeGeneratedFunctions
using LoopVectorization, Primes, BenchmarkTools

#RuntimeGeneratedFunctions.init(@__MODULE__)

# Module to store all runtime-generated FFT functions
module FFTStorage
    using RuntimeGeneratedFunctions
    using ..RadixFFTCompositor

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

    @inline function D(p, m, ::Type{T})::AbstractMatrix where T <: AbstractFloat
        w = cispi.(T(-2/(p*m)) * collect(1:m-1))
        d = zeros(Complex{T},(p-1)*(m-1))

        @inbounds d[1:m-1] .= w

        @inbounds @simd for j in 2:p-1
              @views d[(j-1)*(m-1)+1:j*(m-1)] .= w .* view(d, (j-2)*(m-1)+1:(j-1)*(m-1))
        end

        return reshape(d, m-1, p-1)
    end
    
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

    @inline function fft_cache_oblivious!(Y::AbstractVector{Complex{T}}, X::AbstractVector{Complex{T}}) where T <: AbstractFloat
    n = length(X)
    
    # If no specialized implementation, use the cache-oblivious approach
    if n <= 8  # Base case
        return fft_direct!(X)
    end
    
    # Factor n into n1 * n2 where n1 ≈ n2 ≈ √n
    n1, n2 = find_closest_factors(n)
    
    # Step 1: View as matrix and transpose
    A = reshape(X, n1, n2)
    MatrixOperations.transpose!(A, n1, n2)
    
    # Step 2: Compute n2 FFTs of size n1
    @inbounds @simd for i in 1:n2
        fft_cache_oblivious!(A[:,i])
    end
    
    # Step 3: Apply twiddle factors
    @inbounds @simd for i in 0:n1-1
        @inbounds @simd for j in 0:n2-1
            A[i+1,j+1] *= D[i, j]
        end
    end
    
    # Step 4: Transpose again
    MatrixOperations.transpose!(A, n1, n2)
    
    # Step 5: Compute n1 FFTs of size n2
    @inbounds @simd for i in 1:n1
        fft_cache_oblivious!(A[:,i])
    end
    
    # Step 6: Final transpose
    transpose_inplace!(A)
    
    return X
end
end
module MatrixOperations

using LoopVectorization

# Vectorized in-place transpose for small matrices
function transpose_small!(a::AbstractMatrix{T}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:i-1
            tmp = a[i,j]
            a[i,j] = a[j,i]
            a[j,i] = tmp
        end
    end
end

# Cache-oblivious transpose with vectorized base case
function transpose!(a::AbstractMatrix{T}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 32
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
@inline function matmul_small!(a::AbstractMatrix{T}, b::AbstractMatrix{T}, c::AbstractMatrix{T}, n::Int) where T <: AbstractFloat
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
function matmul!(a::AbstractMatrix{T}, b::AbstractMatrix{T}, c::AbstractMatrix{T}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 32
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
@inline function strassen_add!(c::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] + b[i,j]
        end
    end
end

@inline function strassen_sub!(c::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] - b[i,j]
        end
    end
end

function strassen_mul!(c::AbstractMatrix{T}, a::AbstractMatrix{T}, b::AbstractMatrix{T}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 64  # Use standard multiplication for small matrices
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

using FFTW, BenchmarkTools

n = 2^6
x= [ComplexF64(i,i) for i in 1:n];
y=  similar(x);
fft_func = FFTStorage.get_or_create_fft(n, Float64)
fft_func(y, x)
F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)

@assert y ≈ F * x
factory_b = @benchmark fft_func($y, $x)
fftw_b = @benchmark $F * $x

println("Custom FFT function benchmark:")
display(factory_b)  # Use `display` for detailed results

println("\nFFTW benchmark:")
display(fftw_b)
