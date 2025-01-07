
# Courtesy of Nikos Pitsianis

using BenchmarkTools, LinearAlgebra, FFTW

function inccounter()
  let counter = 0
    return () -> begin
      counter += 1
      return counter
    end
  end
end

inc = inccounter()

function recfft(y, x, w=nothing)
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
    wn = exp(-2im * π / n) .^ (0:(n2)-1)
    s1 = recfft(t[1:n2], x[1:2:n])
    s2 = recfft(t[n2+1:n], x[2:2:n], wn)

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

function makefftradix(n)

  x = map(i -> "x[$i]", 1:n)
  y = map(i -> "y[$i]", 1:n)
  s = recfft(y, x)
  txt = """
  @inbounds function fft_radix$(n)!(y,x)
    $s
    nothing
  end
  """

  #show(txt)
  return eval(Meta.parse(txt))
end

function fft_recursive(x)
  n = length(x)
  if n == 1
    return x
  end

  w = exp(-2im * π / n) .^ (0:(n÷2-1))

  # Compute FFT of even and odd indexed elements
  even_fft = fft_recursive(x[1:2:end])
  odd_fft = w .* fft_recursive(x[2:2:end])

  return vcat(even_fft + odd_fft, even_fft - odd_fft)
end

function testfft(n)
  x = rand(ComplexF64,n)
  y = similar(x)
  f = eval(Symbol("fft_radix$(n)!"))
  f(y,x)
  @assert y ≈ fft(x)
  println("Generated function: \n")
  display(@benchmark $f($y,$x))
  println("FFTW: \n")
  F = FFTW.plan_fft(x; flags=FFTW.PATIENT)
  display(@benchmark $F * $x)
  nothing
end

for q in 1:10
  n = 2^q
  println("Testing n = $n")
  makefftradix(n)
  testfft(n)
end

#=
using BenchmarkTools, LinearAlgebra, FFTW

function inccounter()
    let counter = 0
        return () -> begin
            counter += 1
            return counter
        end
    end
end

"""
Generate symbolic representation of twiddle factors
"""
function gen_twiddle_expression(k::Int, j::Int, m::Int)
    angle = -2π * j * k / m
    real_part = round(cos(angle), digits=10)
    imag_part = round(sin(angle), digits=10)
    
    if real_part == 1 && imag_part == 0
        return ""
    elseif real_part == -1 && imag_part == 0
        return "-"
    else
        return "($(real_part) + $(imag_part)im)*"
    end
end

"""
Generate a single butterfly stage
"""
function gen_butterfly_stage(y_indices, x_indices, m::Int)
    ops = String[]
    
    for k in 0:(m-1)
        terms = String[]
        for j in 0:(m-1)
            if j < length(x_indices)
                twiddle = gen_twiddle_expression(k, j, m)
                push!(terms, "$(twiddle)$(x_indices[j+1])")
            end
        end
        if k < length(y_indices)
            push!(ops, "$(y_indices[k+1]) = " * join(filter(!isempty, terms), " + "))
        end
    end
    
    return join(ops, "\n")
end

"""
Recursive FFT code generator for radix-m
"""
function recfft_m(y_indices, x_indices, m::Int)
    n = length(x_indices)
    
    if n == 1
        return "$(y_indices[1]) = $(x_indices[1])"
    end
    
    if n ≤ m
        return gen_butterfly_stage(y_indices, x_indices, n)
    end
    
    inc = inccounter()
    stage_size = n ÷ m
    temp_vars = [["t$(inc())_$(j)" for j in 1:stage_size] for _ in 1:m]
    
    # Generate recursive stages
    substages = String[]
    for i in 0:(m-1)
        subindices = x_indices[i+1:m:n]
        if !isempty(subindices)
            push!(substages, recfft_m(temp_vars[i+1], subindices, m))
        end
    end
    
    # Generate final butterfly
    final_stage = gen_butterfly_stage(y_indices, vcat(temp_vars...), m)
    
    # Combine all stages
    return """
    # Compute sub-FFTs
    $(join(substages, "\n"))
    
    # Final butterfly stage
    $final_stage
    """
end

"""
Generate complete FFT function for radix-m
"""
function make_fft_radix_m(n::Int, m::Int)
    @assert n > 0 "Size must be positive"
    @assert m > 0 "Radix must be positive"
    @assert mod(n, m) == 0 "Size must be divisible by radix"
    
    # Generate input and output variable names
    x_vars = ["x[$i]" for i in 1:n]
    y_vars = ["y[$i]" for i in 1:n]
    
    # Generate the FFT code
    fft_code = recfft_m(y_vars, x_vars, m)
    
    # Create temporary variable declarations
    temp_vars = []
    for i in 1:inccounter()()-1
        push!(temp_vars, "t$(i)_1 = zero(ComplexF64)")
    end
    
    # Create the complete function
    func_def = """
    function fft_radix$(m)_$(n)!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64})
        # Temporary variables
        $(join(temp_vars, "\n    "))
        
        # FFT computation
        $fft_code
        
        return nothing
    end
    """
    
    # Evaluate and return the function
    return eval(Meta.parse(func_def))
end

"""
Test function for radix-m FFT
"""
function test_fft_radix_m(n::Int, m::Int)
    println("\nGenerating and testing radix-$m FFT for size $n")
    
    try
        # Generate the FFT function
        f = make_fft_radix_m(n, m)
        
        # Create test data
        x = rand(ComplexF64, n)
        y = zeros(ComplexF64, n)
        
        # Run the generated FFT
        f(y, x)
        
        # Compare with FFTW
        reference = fft(x)
        error = maximum(abs.(y - reference))
        
        println("Maximum error vs FFTW: $error")
        
        if error < 1e-10
            println("Test passed!")
            
            # Benchmark
            println("\nBenchmarking generated FFT:")
            display(@benchmark $f($y, $x))
            
            println("\nBenchmarking FFTW:")
            F = FFTW.plan_fft(x)
            display(@benchmark $F * $x)
        else
            println("Test failed: Error too large")
        end
        
    catch e
        println("Error in FFT generation/testing:")
        println(e)
        return nothing
    end
end

# Example usage and testing
function test_various_radixes()
    test_cases = [
        (9, 3),    # Radix-3
        (16, 4),   # Radix-4
        (27, 3),   # Radix-3
        (32, 2),   # Radix-2
        (81, 3)    # Radix-3
    ]
    
    for (n, m) in test_cases
        test_fft_radix_m(n, m)
    end
end

# Export main functions
export make_fft_radix_m, test_fft_radix_m, test_various_radixes

test_various_radixes()
=#