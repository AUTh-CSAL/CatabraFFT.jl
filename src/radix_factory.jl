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
            angle_cos = T(cospi(angle))
            angle_sin = T(sinpi(angle))
            
            # Only store non-trivial values (not 0, 1, -1)
            #if !isapprox(abs(angle_cos), 1.0) && !isapprox(abs(angle_cos), 0.0)
                str *= "const CP_$(n4-i)_$(n2) = $angle_cos\n"
            #end
            #if !isapprox(abs(angle_sin), 1.0) && !isapprox(abs(angle_sin), 0.0)
                str *= "const SP_$(n4-i)_$(n2) = $angle_sin\n"
            #end
        end
        str *= "\n"
        current_n >>= 1
    end
    
    # Add only essential special constants
    if n >= 8
        str *= "const INV_SQRT2 = $(T(1/sqrt(2)))\n"
    end
    
    @show str
    return str
end

"""
Generate constant expressions for a given radix-2^s size
"""
function generate_radix_constants(s::Integer)::Vector{Pair{Complex{Float64}, String}}
    n = 2^s
    constants = Pair{Complex{Float64}, String}[]
    
    for k in 0:(n-1)
        angle = -2π * k / n
        w = cispi(-2 * k / n)
        
        # Generate constant name based on angle
        constant_expr = get_constant_expression(w, n)
        push!(constants, w => constant_expr)
    end
    
    sort!(constants, by=x->abs(angle(x.first)))  # Sort by angle for consistent ordering
    return constants
end

"""
Enhanced twiddle factor expression generator with improved constant recognition
"""
function get_constant_expression(w::Complex{T}, n::Integer)::String where T <: AbstractFloat
    real_part = real(w)
    imag_part = imag(w)
    
    # Helper for approximate equality
    isclose(a, b) = abs(a - b) < eps(T) * 10
    
    # Function to get sign string
    sign_str(x) = x ≥ 0 ? "+" : "-"
    
    # Common cases table with twiddle factors patterns commonly met

    common_cases = [
        (1.0, 0.0) => "1",
        (-1.0, 0.0) => "-1",
        (0.0, 1.0) => "im",
        (0.0, -1.0) => "-im",
        (1/√2, 1/√2) => "INV_SQRT2*(1+im)",
        (1/√2, -1/√2) => "INV_SQRT2*(1-im)",
        (-1/√2, 1/√2) => "INV_SQRT2*(-1+im)",
        (-1/√2, -1/√2) => "-INV_SQRT2*(1+im)"
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
        @show angles
        for (num, den) in angles
            cp = cospi(num/den)
            sp = sinpi(num/den)
            if isclose(abs(real_part), cp) && isclose(abs(imag_part), sp)
                return "$(sign_str(real_part))CP_$(num)_$(den) $(sign_str(imag_part))SP_$(num)_$(den)*im"
            elseif isclose(abs(real_part), sp) && isclose(abs(imag_part), cp)
                return "$(sign_str(real_part))SP_$(num)_$(den) $(sign_str(imag_part))CP_$(num)_$(den)*im"
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
    # println("*** n = $n")
    t = map(i -> "t$(inc())", 1:n)
    n2 = n ÷ 2
    wn = get_twiddle_expression(collect(0:n2-1), n)

    s1 = recfft2(t[1:n2], x[1:2:n])
    s2 = recfft2(t[n2+1:n], x[2:2:n], wn)

    if isnothing(w)
      s3p = foldl(*, map(i -> ",$(y[i])", 2:n2); init="$(y[1])") *
            " = " *
            foldl(*, map(i -> ",$(t[i]) + $(t[i+n2])", 2:n2), init="$(t[1]) + $(t[1+n2])") * "\n"
      s3m = foldl(*, map(i -> ",$(y[i+n2])", 2:n2); init="$(y[n2+1])") *
            " = " *
            foldl(*, map(i -> ",$(t[i]) - $(t[i+n2])", 2:n2), init="$(t[1]) - $(t[1+n2])") * "\n"
    else
      s3p = foldl(*, map(i -> ", $(y[i])", 2:n2); init="$(y[1])") *
            " = " *
            foldl(*, map(i -> ", ($(w[i]))*($(t[i]) + $(t[i+n2]))", 2:n2), init="($(w[1]))*($(t[1]) + $(t[1+n2]))") * "\n"
      s3m = foldl(*, map(i -> ", $(y[i+n2])", 2:n2); init="$(y[n2+1])") *
            " = " *
            foldl(*, map(i -> ", ($(w[n2+i]))*($(t[i]) - $(t[i+n2]))", 2:n2), init="($(w[n2+1]))*($(t[1]) - $(t[1+n2]))") * "\n"
    end

    return s1 * s2 * s3p * s3m
  end
end

function makefftradix(n, ::Type{T}) where T <: AbstractFloat

  x = map(i -> "#INPUT#[$i]", 1:n)
  y = map(i -> "#OUTPUT#[$i]", 1:n)
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
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int)"
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
            INV_SQRT2 = T(INV_SQRT2_DEFAULT)
            $decorators for q in 1:s
                $kernel_code
            end
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
    for i in 1:radix-1
        twiddle_expression = i == 1 ? "w1p" : "w$(div(i, 2))p * w$(div(i + 1, 2))p"
        push!(twiddle_code, "w$(i)p = $twiddle_expression")
    end
    twiddle_code_str = join(twiddle_code, "\n")
    
    return """
    @inline function $name$signature

        #TODO ADD INIT KERNEL

        # Section with twiddle factors
        $decorators for p in 1:(n1-1)
            w1p = cispi(T(-p * theta))
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
        ["y"],
        #["y", "ivdep"],
        ["layered"],
        #["layered", "ivdep"]
    ]
    
    kernels = Dict{String, String}()
    
    for radix in radices
        for suffixes in suffix_combinations
            name = generate_kernel_name(radix, suffixes)
            code = generate_kernel(radix, suffixes, T)
            kernels[name] = code
        end
    end
    
    @show kernels
    
    return kernels
end

# Function to evaluate and create the functions in a module
function create_kernel_module(N::Int, ::Type{T}) where T <: AbstractFloat
    module_constants = generate_module_constants(N, T)
    @show module_constants
    kernels = generate_all_kernels(N, T)
    
    family_module_code = """
    module radix_2_family
        using LoopVectorization
        
        $module_constants
        
        $(join(values(kernels), "\n\n"))
    end
    """
    
    return family_module_code
end


end
RadixFFTCompositor.create_kernel_module(128, Float32)