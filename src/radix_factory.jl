module RadixGenerator

include("helper_tools.jl")

using LoopVectorization, GeneralizedGenerated

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

        # Add the extract_view lamda in module definition for layered kernels
        #str *= "extract_view = (x::Vector{Complex{$T}}, q::Int, p::Int, s::Int, n1::Int, N::Int) -> [x[q + s*(p + i*n1)] for i in 0:N-1]"
        str *= "extract_view = (x::Vector{Complex{$T}}, q::Int, p::Int, s::Int, n1::Int, N::Int) -> view(x, q .+ s*(p .+ (0:(N-1))*n1))"
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
        (1.0, 0.0) => "1", # TODO SKIP UNITARY MULTIPLICATIONS
        (-1.0, 0.0) => "-1",
        (0.0, 1.0) => "im",
        (0.0, -1.0) => "-im",
        (1/√2, 1/√2) => "INV_SQRT2_Q1",
        (1/√2, -1/√2) => "INV_SQRT2_Q4",
        (-1/√2, 1/√2) => "-INV_SQRT2_Q4",
        (-1/√2, -1/√2) => "-INV_SQRT2_Q1"
    ]
    
    # Check special cases fist
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
    
    # Fallback to numerical representation with high precision if everything else fails
    return "($(round(real_part, digits=16))$(sign_str(imag_part))$(abs(round(imag_part, digits=16)))*im)"
end

"""
Generate twiddle factor expressions for a given collection of indices
"""
function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

# 'recfft2' logic is courtesy of Nikos Pitsianis 
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

# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int, suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    input = "y" ∈ suffixes ? "y" : "x"
    output = "y"

    x = ["$input[$i]" for i in 1:n]
    y = ["$output[$i]" for i in 1:n]

    s = recfft2(y, x) # Replace with any other recfft kernel family seed.
    
    kernel_code = replace(s, 
            "#INPUT#" => input,
            "#OUTPUT#" => output)

    return kernel_code
end

# Function to generate kernel name
function generate_kernel_names(radix::Int, suffixes::Vector{String})
    base = "fft$(radix)_shell"
    suffix = join(suffixes, "_")
    return (string(base, isempty(suffix) ? "" : "_$suffix", "!"), string(base, "!"))
    #return (string(base, isempty(suffix) ? "" : "$suffix", "!"), string(base, "!"))
end

# Function to generate function signature
function generate_signature(suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    y_only = "y" in suffixes
    layered = "layered" in suffixes
    
    if y_only
        return "(y::AbstractVector{Complex{$T}})"
    elseif layered
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int, n1::Int, theta::$T=$T(0.125))"
    else
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})"
    end
end

# Main function to generate kernel code
function generate_kernel(radix::Int, suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    names = generate_kernel_names(radix, suffixes)
    signature = generate_signature(suffixes, T)
    
    kernel_code = makefftradix(radix, suffixes, T)
    
    # Generate the complete function
    if "layered" ∈ suffixes
        # Special handling for layered kernels
        return generate_layered_kernel(radix, names, signature, suffixes)
    else
        return """
        @inline function $(names[2])$signature 
            $kernel_code
        end
        """
    end
end

# Helper function to generate layered kernel
function generate_layered_kernel(radix, names, signature, suffixes)
    s = log2(radix)  # Assuming radix is a power of 2
    @assert isinteger(s) && s > 0 "Radix must be a power of 2"

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

    decorators = generate_loop_decorators(suffixes)
    
    # Generate twiddle factor computation dynamically for any radix
    twiddle_code = String[]
    for i in 2:radix-1
        twiddle_expression = "w$(div(i, 2))p * w$(div(i + 1, 2))p"
        push!(twiddle_code, "w$(i)p = $twiddle_expression")
    end
    twiddle_code_str = join(twiddle_code, "\n")

    # Generate layer_y twiddle factor application
    twiddle_apply_code = ["""layer_y[$i] *= w$(i-1)p""" for i in 2:radix]
    #twiddle_apply_code = ["""layer_y[$i] = dot(wp[1:$(radix-1)], layer_y[2:$radix])""" for i in 2:radix]
    twiddle_apply_str = join(twiddle_apply_code, "\n    ")
    
    return """
    @inline function $(names[1])$signature

        @inbounds @simd for q in 1:s
        layer_x, layer_y = extract_view(x, q, 0, s, n1, $radix), extract_view(y, q, 0, s, n1, $radix)
        $(names[2])(layer_y, layer_x)
        end

        # Section with twiddle factors
        $decorators for p in 1:(n1-1)
            w1p = cispi(-p * theta)
            $twiddle_code_str
            $decorators for q in 1:s
                layer_x, layer_y = extract_view(x, q, p, s, n1, $radix), extract_view(y, q, p, s, n1, $radix)
                $(names[2])(layer_y, layer_x)
                $twiddle_apply_str
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
        
    radices = subpowers_of_two(N)

    suffix_combinations = Vector{Vector{String}}([
        String[],
        #["ivdep"],
        #["y"],
        #["y", "ivdep"],
        ["layered"], # Must have generated normal kernels to produces functional layered kernels
        #["layered", "ivdep"]
    ])
    
    kernels = Vector{String}()
    
    for radix ∈ radices
        for suffixes ∈ suffix_combinations
            push!(kernels, generate_kernel(radix, suffixes, T))
        end
    end
    
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
    
    return Meta.parse(family_module_code)  # Parse directly into an expression
end

function evaluate_fft_generated_module(target_module::Module, n::Int, ::Type{T}) where T <: AbstractFloat
    module_expr = create_kernel_module(n, T)
    Core.eval(target_module, module_expr)
end

end

#=
module Testing
using FFTW, BenchmarkTools
using ..RadixGenerator

function main()
n = 2^7
Type = Float64
RadixGenerator.evaluate_fft_generated_module(Testing, n, Type)
x = [Complex{Type}(i,i) for i in 1:n]
y = similar(x)
Base.invokelatest(radix_2_family.fft128_shell!, y, x)
#@show y

F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)
y_fftw = F * x
#@show y_fftw
##@show y_fftw
@assert y_fftw ≈ y
println("Done")

b_fftgen = @benchmark radix_2_family.fft128_shell!($y, $x)
b_fftw = @benchmark $F * $x
println("Display Generated FFT:")
display(b_fftgen)

println("Display FFTW:")
display(b_fftw)

end
end

Testing.main()
=#