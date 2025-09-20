include("suffix.jl")
using BenchmarkTools, SIMD

"""
# Usage examples:
load_real_imag_gen(["x1", "x2"], mode=:default)
load_real_imag_gen(["x1", "x2"], mode=:unsafe_load, ptr_name="data_ptr")
load_real_imag_gen(["x1", "x2"], mode=:vload_soa, vec_width=8)
"""
load_real_imag_gen = (t; mode, T, ptr_name="px") -> begin
    vec_width = 2sizeof(T)  # Width in bytes for real+imag pair
    
    if mode == :unsafe_load
        # Pointer-based scalar loads (e.g. xmm SSE4 registers)
        # For small kernels utilizing Instruction-Level Parallelism (ILP)
        # Modern CPUs can execute multiple independent scalar operations simultaneously:
        # Can execute in parallel on different execution ports
        # If your CPU likely has 2-3 ADD units, these execute in parallel despite being "scalar".
        # No SIMD Setup Overhead! Shuffling data into SIMD layout, permuting for butterfly patterns and extracting results
        # ...can be MORE expensive than simple scalar ops!
        # -> unsafe_store!(py, #, i) uses ymm register however...
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                idx_r = 2*num - 1
                idx_i = 2*num
                "$(prefix)$(var)$(num)_r, $(prefix)$(var)$(num)_i = unsafe_load($ptr_name, $idx_r), unsafe_load($ptr_name, $idx_i)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
        
    elseif mode == :vgather
        # SIMD vgather for strided access patterns
        avx2_bits = 256
        
        # Calculate how many complex numbers fit in a ymm register
        complex_size_bits = 2 * sizeof(T) * 8  # 2 floats per complex * bytes * 8 bits/byte
        complexes_per_vec = avx2_bits ÷ complex_size_bits
        
        # Number of elements to process
        n_elems = length(t)
        
        if n_elems <= complexes_per_vec
            # All elements fit in one SIMD register
            # Generate gather indices for interleaved real/imag layout
            indices = Int[]
            vars = String[]
            
            for (i, s) in enumerate(t)
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                
                # For complex array stored as [r1,i1,r2,i2,r3,i3,...]
                # We need indices for both real and imaginary parts
                push!(indices, 2*num - 1)  # real index (1-based)
                push!(indices, 2*num)      # imag index (1-based)
                push!(vars, var)
            end
            
            # Generate the vgather code
            code_parts = String[]
            
            # Create index vector for gathering
            idx_tuple = "(" * join(indices, ",") * ")"
            push!(code_parts, "idx = Vec{$(2*n_elems),Int64}($idx_tuple)")
            
            # Perform the gather
            push!(code_parts, "v = vgather($(ptr_name), idx)")
            
            # Extract real and imaginary parts from the gathered vector
      #=
            for (i, s) in enumerate(t)
                m = match(r"(\d+)\D*$", s)
                num = m.captures[1]
                var = vars[i]
                
                # Extract from gathered vector (0-based indexing for getindex)
                real_idx = 2*i - 2  # 0-based index for real part
                imag_idx = 2*i - 1  # 0-based index for imag part
                
                push!(code_parts, "$(var)$(num)_r = v[$(real_idx + 1)]")
                push!(code_parts, "$(var)$(num)_i = v[$(imag_idx + 1)]")
            end
      =#
            
            join(code_parts, "\n    ")
            
        else
            # Need multiple SIMD loads - process in chunks
            code_parts = String[]
            
            # Process elements in groups that fit in ymm registers
            for chunk_start in 1:complexes_per_vec:n_elems
                chunk_end = min(chunk_start + complexes_per_vec - 1, n_elems)
                chunk_size = chunk_end - chunk_start + 1
                
                indices = Int[]
                chunk_vars = String[]
                
                for i in chunk_start:chunk_end
                    s = t[i]
                    m = match(r"(\d+)\D*$", s)
                    num = parse(Int, m.captures[1])
                    var = startswith(s, "x") ? "x" :
                          startswith(s, "y") ? "y" :
                          startswith(s, "D") ? "d" : error("Unknown input: $s")
                    
                    push!(indices, 2*num - 1)  # real index
                    push!(indices, 2*num)      # imag index
                    push!(chunk_vars, "$(var)$(num)")
                end
                
                # Pad indices if needed for full vector width
                while length(indices) < 2*complexes_per_vec
                    push!(indices, 1)  # Pad with valid index (will be ignored)
                end
                
                idx_tuple = "(" * join(indices[1:2*complexes_per_vec], ",") * ")"
                chunk_id = (chunk_start - 1) ÷ complexes_per_vec + 1
                
                push!(code_parts, "idx$(chunk_id) = Vec{$(2*complexes_per_vec),Int64}($idx_tuple)")
                push!(code_parts, "v$(chunk_id) = vgather($(ptr_name), idx$(chunk_id))")
                
                # Extract values
        #=
                for (j, var_num) in enumerate(chunk_vars)
                    real_idx = 2*j - 1  # 1-based index in vector
                    imag_idx = 2*j      # 1-based index in vector
                    push!(code_parts, "$(var_num)_r = v$(chunk_id)[$(real_idx)]")
                    push!(code_parts, "$(var_num)_i = v$(chunk_id)[$(imag_idx)]")
                end
      =#
            end
            
            join(code_parts, "\n    ")
        end
        
    elseif mode == :vload_soa
        # Structure of Arrays - existing implementation
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                "$(prefix)$(var)$(num) = vload(Vec{$vec_width, $T}, $ptr_name + 2*$(num-1)*$vec_width)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
        
    elseif mode == :vload_aos
        # Array of Structures - existing implementation
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                offset = (num-1)*2*vec_width
                "$(prefix)$(var)$(num)_ri = vload(Vec{$(2*vec_width), $T}, $ptr_name, $offset + 1)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
        
    else
        # Default: original behavior
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = m.captures[1]
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                rhs = occursin('[', s) ? replace(s, " " => "") : "$var[$num]"
                prefix = i == 1 ? "" : " "
                "$(prefix)$(var)$(num)_r , $(prefix)$(var)$(num)_i = real($rhs), imag($rhs)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
    end
end

function generate_var_names(group_size::Int)
    # Generate xa, xb, xc, ... based on group_size
    return ["x" * Char('a' + i) for i in 0:group_size-1]
end

function replace_with_reused_vars(s::String, group_size=4)
    vars = generate_var_names(group_size)
    n_vars = length(vars)
    
    # Find the maximum x number in the string
    max_num = 0
    for m in eachmatch(r"x(\d+)", s)
        max_num = max(max_num, parse(Int, m.captures[1]))
    end
    
    result = s
    # Process in descending order
    for num in max_num:-1:1
        old = "x$num"
        group_idx = ((num - 1) ÷ group_size) % n_vars
        new = vars[group_idx + 1]
        
        # Use negative lookahead to ensure we don't match x1 in x13
        result = replace(result, Regex("$(old)(?!\\d)") => new)
    end
    
    return result
end


# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int,  suffixes::SuffixFlags, D::AbstractArray{String}, p::Int, s::Int, SIZE::Int, ::Type{T}) where T <: AbstractFloat

  global inc = inccounter() # nullify global tmp 't' var counter for each new kernel generated
  
  mode = :vgather

  has_y = has_flag(suffixes, Y)
  has_mat = has_flag(suffixes, MAT)
  has_vec = has_flag(suffixes, VEC)
  input = has_y ? "y" : "x"
  output = "y"
  groups = SIZE ÷ s

  prev_output = output
  unsafe_load_mode = mode == :unsafe_load
  if unsafe_load_mode
    output *= "_floats"
  end
  
  if has_mat
      if has_vec
        x = ["$(input)$(i + p*s)" for i in 1:n]
        y = ["$output[$(i + p*s)]" for i in 1:n]
      else
        if unsafe_load_mode
          x = ["$(input)[$(2*(p + 1 + (i-1)*groups) - 1 + j)]" for i in 1:n for j in 0:1]
          y = ["$(output)[$(2*(i + p*s) - 1 + j)]" for i in 1:n for j in 0:1]
        else
          x = ["$(input)$(p + 1 + (i-1)*groups)" for i in 1:n]
          y = ["$output[$(i + p*s)]" for i in 1:n]
        end
      end
      d = D == String[] ? nothing : D
  else
    if has_vec
      x = ["$(input)($i + offset)" for i in 1:n]
      y = ["$output[$i + offset]" for i in 1:n]
    else
      if unsafe_load_mode
        x = ["$(input)$i" for i in 1:s:n]
        y = ["$output[$i]" for i in 1:2n]
      else
        x = ["$(input)$i" for i in 1:s:n]
        y = ["$output[$i]" for i in 1:n]
      end
    end
    d = nothing
  end

  # Generate kernel code as string first
  px = if (mode == :unsafe_load) "p$(input) = pointer(reinterpret($T, $(input)));"
      elseif (mode == :vgather) "p$(input) = complex_to_float_zerocopy($(input));"
      else ""
      end
  py = (mode == :unsafe_load) ? "$(output) = reinterpret($T, $(prev_output));" : ""
  kernel_code = recfft2(y, x, d, nothing, true, T, 1, mode, py) 
  kernel_code = "$px" * "\n" * kernel_code
    
  
  # Parse the string into actual Julia expressions
  if isempty(kernel_code)
      return quote end
  else
      try
          # Wrap in begin...end block for parsing multiple statements
          parsed_expr = Meta.parse("begin\n$kernel_code\nend")
          @show parsed_expr
          return parsed_expr
      catch e
          @warn "Failed to parse kernel code: $e"
          @warn "Kernel code was: $kernel_code"
          # Return a fallback expression
          return quote
              copyto!(y, x)
          end
      end
  end
end

function parse_x(s::String)
  pattern = r"\d+"
  m = match(pattern, s)
  return m !== nothing ? parse(Int, m.match) : nothing
end

parse_x(arr::AbstractArray{String}) = parse_x.(arr)

function map_to_groups(numbers::AbstractArray{Int}, MODULO::Int)
  return ((numbers .- 1) .÷ MODULO) .+ 1
end

function parse_cispi(s::String)
    # Enhanced regex pattern with optional sign and im* prefix
    pattern = r"^([+-]?)(im\*)?CISPI_(\d+)_(\d+)_Q([14])$"
    
    m = match(pattern, s)
    isnothing(m) && error("Invalid CISPI format: $s")

    # Extract components with new prefix handling
    num = parse(Int, m[3])
    den = parse(Int, m[4])
    is_q1 = m[5] == "1"

    return (num=num, den=den, q1=is_q1)
end

parse_cispi(arr::AbstractArray{String}) = parse_cispi.(arr)

function add_more_tmp_vars(x1, x2, wn, n)
    tmp_vars = String[]
    assignments = String[]
    idx = 0

    for i in 1:n
      #if wn[i] ∉ ("1", "-im")
        real_plus = x1[2*i - 1]
        imag_plus = x1[2*i]
        push!(tmp_vars, "tmp$(idx)_r", "tmp$(idx)_i")
        push!(assignments, real_plus, imag_plus)
        idx += 1
      #end
    end

    for i in 1:n
      #if wn[i] ∉ ("1", "-im")
        real_minus = x2[2*i - 1]
        imag_minus = x2[2*i]
        push!(tmp_vars, "tmp$(idx)_r", "tmp$(idx)_i")
        push!(assignments, real_minus, imag_minus)
        idx += 1
      #end
    end

    if !isempty(tmp_vars)
      return "$(join(tmp_vars, ", ")) = $(join(assignments, ", "))\n"
    end

    return ""
end

function sat_expr(tmp, w)
    if w == "1"
        return "$(tmp)_r, $(tmp)_i"
    elseif w == "-im"
        return "$(tmp)_i, $(tmp)_r"
    elseif w == "INV_SQRT2_Q4"
        # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
        return "INV_SQRT2*($(tmp)_r + $(tmp)_i), " *
               "INV_SQRT2*($(tmp)_i - $(tmp)_r)"
    elseif w == "-INV_SQRT2_Q1"
        # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ± b_i)/√2 ]
        return "INV_SQRT2*($(tmp)_i - $(tmp)_r), " *
               "-INV_SQRT2*($(tmp)_r + $(tmp)_i)"
    else
        num, den, is_q1 = parse_cispi(w)
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"
        
        if startswith(w, "CISPI")
            return is_q1 ?
                # Q1: cosθ + i sinθ
                "muladd($c, $(tmp)_r, -$s * $(tmp)_i), " *
                "muladd($s, $(tmp)_r, $c * $(tmp)_i)" :
                # Q4: cosθ - i sinθ
                "muladd($c, $(tmp)_r , $s * $(tmp)_i), " *
                "muladd(-$s, $(tmp)_r, $c * $(tmp)_i) "
        
        elseif startswith(w, "-im*CISPI")
            return is_q1 ?
                # -i*(cosθ + i sinθ) = sinθ - i cosθ
                "muladd($s, $(tmp)_r, $c * $(tmp)_i), " *
                "muladd(-$c, $(tmp)_r, $s * $(tmp)_i)" :
                # -i*(cosθ - i sinθ) = -sinθ - i cosθ
                "muladd(-$s, $(tmp)_r, $c * $(tmp)_i), " *
                "muladd(-$c, $(tmp)_r, -$s * $(tmp)_i)"
        
        elseif startswith(w, "-CISPI")
            return is_q1 ?
                # -cosθ - i sinθ
                "muladd(-$c, $(tmp)_r, $s * $(tmp)_i), " *
                "muladd(-$s, $(tmp)_r, -$c * $(tmp)_i)" :
                # -cosθ + i sinθ
                "muladd(-$c, $(tmp)_r, -$s * $(tmp)_i), " *
                "muladd($s, $(tmp)_r, -$c * $(tmp)_i)"

        elseif startswith(w, "im*CISPI")
            return is_q1 ?
                # i*(cosθ + i sinθ) = -sinθ + i cosθ
                "muladd(-$s, $(tmp)_r, -$c * $(tmp)_i), " *
                "muladd($c, $(tmp)_r, -$s * $(tmp)_i)" :
                # i*(cosθ - i sinθ) = sinθ + i cosθ
                "muladd($s, $(tmp)_r, -$c * $(tmp)_i), " *
                "muladd($c, $(tmp)_r, $s * $(tmp)_i)"
        end
    end
end

function sat_expr(sign, x1, x2, w)
  is_t = startswith(x1, "t") || startswith(x2, "t")
  if w == "1"
      #return is_t ? 
          "$(x1)_r $sign $(x2)_r, $(x1)_i $sign $(x2)_i" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
  elseif w == "-im"
      # -i*(a ± b) = ±(b_i ∓ a_i) ± i*(b_r ∓ a_r)
      #return is_t ? 
          "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
  elseif w == "INV_SQRT2_Q4"
      # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
      #return is_t ?
          "INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i)), " *
          "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r))" #:
          #"INV_SQRT2*(($(x1)[1] $sign $(x2)[1]) + ($(x1)[2] $sign $(x2)[2])), " *
          #"INV_SQRT2*(($(x1)[2] $sign $(x2)[2]) - ($(x1)[1] $sign $(x2)[1]))"
  elseif w == "-INV_SQRT2_Q1"
      # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ± b_i)/√2 ]
      #return is_t ? 
          "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r)), " *
          "-INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i))" #:
          #"INV_SQRT2*(($(x1)[2] $sign $(x2)[2]) - ($(x1)[1] $sign $(x2)[1])), " *
          #"-INV_SQRT2*(($(x1)[1] $sign $(x2)[1]) + ($(x1)[2] $sign $(x2)[2]))"
  else
      num, den, is_q1 = parse_cispi(w)
      c = "COSPI_$(num)_$(den)"
      s = "SINPI_$(num)_$(den)"
      
      if startswith(w, "CISPI")
          if is_q1
              # Q1: cosθ + i sinθ
              #return is_t ?
              "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
              "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd($c, $(x1)[1] $sign $(x2)[1], -$s * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd($s, $(x1)[1] $sign $(x2)[1], $c * ($(x1)[2] $sign $(x2)[2]))"
          else
              # Q4: cosθ - i sinθ
              #return is_t ?
              "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
              "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd($c, $(x1)[1] $sign $(x2)[1], $s * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd(-$s, $(x1)[1] $sign $(x2)[1], $c * ($(x1)[2] $sign $(x2)[2]))"
          end
      
      elseif startswith(w, "-im*CISPI")
          if is_q1 
              # -i*(cosθ + i sinθ) = sinθ - i cosθ
              #return is_t ?
              "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
              "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd($s, $(x1)[1] $sign $(x2)[1], $c * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd(-$c, $(x1)[1] $sign $(x2)[1], $s * ($(x1)[2] $sign $(x2)[2]))" 
          else
              # -i*(cosθ - i sinθ) = -sinθ - i cosθ
              #return is_t ?
              "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
              "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd(-$s, $(x1)[1] $sign $(x2)[1], $c * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd(-$c, $(x1)[1] $sign $(x2)[1], -$s * ($(x1)[2] $sign $(x2)[2]))"
          end
      
      elseif startswith(w, "-CISPI")
          if is_q1 
              # -cosθ - i sinθ
              #return is_t ?
              "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
              "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd(-$c, $(x1)[1] $sign $(x2)[1], $s * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd(-$s, $(x1)[1] $sign $(x2)[1], -$c * ($(x1)[2] $sign $(x2)[2]))" 
          else
              # -cosθ + i sinθ
              #return is_t ?
              "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
              "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd(-$c, $(x1)[1] $sign $(x2)[1], -$s * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd($s, $(x1)[1] $sign $(x2)[1], -$c * ($(x1)[2] $sign $(x2)[2]))"
          end

      elseif startswith(w, "im*CISPI")
          if is_q1 
              # i*(cosθ + i sinθ) = -sinθ + i cosθ
              #return is_t ?
              "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
              "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd(-$s, $(x1)[1] $sign $(x2)[1], -$c * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd($c, $(x1)[1] $sign $(x2)[1], -$s * ($(x1)[2] $sign $(x2)[2]))" 
          else
              # i*(cosθ - i sinθ) = sinθ + i cosθ
              #return is_t ?
              "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
              "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))" #:
              #"muladd($s, $(x1)[1] $sign $(x2)[1], -$c * ($(x1)[2] $sign $(x2)[2])), " *
              #"muladd($c, $(x1)[1] $sign $(x2)[1], $s * ($(x1)[2] $sign $(x2)[2]))"
          end
      end
  end
end

function inccounter()
  let counter = 0
    return () -> (counter += 1)
  end
end

inc = inccounter()

function recfft2(y, x, d, w, root, ::Type{T}, tmp_base=1, mode=:default, py="") where T <: AbstractFloat
  n = length(x)
  MODULO = 4

  if n == 1
    ""
  elseif n == 2
    s = if !isnothing(d)
          if isnothing(w)
            if root
              if mode == :unsafe_load
                load_real_imag_gen(x; mode=mode, T=T) * "\n" * 
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" * """
                $(y[1]), $(y[2]), $(y[3]), $(y[4]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(sat_expr("tmp0", "$(d[1])"))
                """
              else
                load_real_imag_gen(x; mode=mode, T=T) * "\n" * 
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" * """
                $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(sat_expr("tmp0", "$(d[1])")))
                """
              end
            end
          end
        else
          if root
            if mode == :unsafe_load
            load_real_imag_gen(x; mode=mode, T=T) * "\n" * "$py" * "\n" * """
            $(y[1]), $(y[2]), $(y[3]), $(y[4]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            """ 
            else
            load_real_imag_gen(x; mode=mode, T=T) * "\n" * """
            $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i)
            """ 
            end 
          else
            if isnothing(w)
            """
            $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            """
            else
            w[1] == "1" ? 
                """
                $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
                """ :
                """
                $(y[1]), $(y[2]) = $(sat_expr("+", "$(x[1])", "$(x[2])", "$(w[1])")), $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
                """
            end
          end
        end
    s
    return s
  else
    n2 = n ÷ 2
    t = ["t$i" for i in tmp_base:tmp_base + n - 1]
    new_tmp_base = tmp_base + n
    
    # Recursively handle sub-transforms
    s1 = recfft2(t[1:n2], x[1:2:n], nothing, nothing, false, T, new_tmp_base, mode, py)
    s2 = recfft2(t[n2+1:n], x[2:2:n], nothing, get_twiddle_expression(collect(0:n2-1), n), false, T, new_tmp_base, mode, py)

    tmp_decls = if n > 2 
      x1_exprs = String[]
      x2_exprs = String[]
      for i in 2:n2
          push!(x1_exprs, "$(t[i])_r + $(t[i+n2])_r")
          push!(x1_exprs, "$(t[i])_i + $(t[i+n2])_i")
          push!(x2_exprs, "$(t[i])_r - $(t[i+n2])_r")
          push!(x2_exprs, "$(t[i])_i - $(t[i+n2])_i")
      end
      if isnothing(d) && !isnothing(w) 
        add_more_tmp_vars(x1_exprs, x2_exprs, w[2:n2], n2-1)
      elseif isnothing(w) && !isnothing(d)
        add_more_tmp_vars(x1_exprs, x2_exprs, d[2:n2], n2-1)
      end
    else
      ""
    end
    
    # Final layer combining with D matrix twiddles
    if !isnothing(d)
      if isnothing(w)
        if root
          if mode == :unsafe_load
           s3p = "$py" * "\n" * "$(tmp_decls)" * "\n" *
                 "$(y[1])" * foldl(*, vmap(i -> ", $(y[i])", 2:2n2)) *
                 " = " *
                 "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-2)", "$(d[i-1])"))", 2:n2)) * "\n"
           s3m = "$(y[2n2+1])" * foldl(*, vmap(i -> ", $(y[i+2n2])", 2:2n2)) *
                 " = " *
                 "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(d[n2])")))" * foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", "$(d[i+n2-1])"))", 2:n2)) * "\n"
          else
           s3p = "$py" * "\n" * "$(tmp_decls)" * "\n" *
                 "$(y[1])" * foldl(*, vmap(i -> ", $(y[i])", 2:n2)) *
                 " = " *
                 "Complex{$T}($(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(sat_expr("tmp$(i-2)", "$(d[i-1])")))", 2:n2)) * "\n"
           s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ", $(y[i+n2])", 2:n2)) *
                 " = " *
                 "Complex{$T}($(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(d[n2])")))" * foldl(*, vmap(i -> ", Complex{$T}($(sat_expr("tmp$(i-3+n2)", "$(d[i+n2-1])")))", 2:n2)) * "\n"
          end
        end
      end
    else
      if isnothing(w)
        if root 
          if mode == :unsafe_load
          s3p = "$py" * "\n" * "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:2n2)) *
                " = " *
                "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i", 2:n2)) * "\n"
          s3m = "$(y[2n2+1])" * foldl(*, vmap(i -> ",$(y[i+2n2])", 2:2n2)) *
                " = " *
                "$(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i", 2:n2)) * "\n"
          else
          s3p = "$py" * "\n" * "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:n2)) *
                " = " *
                "Complex{$T}($(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i)", 2:n2)) * "\n"
          s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ",$(y[i+n2])", 2:n2)) *
                " = " *
                "Complex{$T}($(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i)", 2:n2)) * "\n"
          end
        else
          s3p = "$(y[1])_r, $(y[1])_i" * foldl(*, vmap(i -> ", $(y[i])_r, $(y[i])_i", 2:n2)) *
                " = " *
                "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i", 2:n2)) * "\n"
          s3m = "$(y[n2+1])_r, $(y[n2+1])_i " * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
                " = " *
                "$(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i", 2:n2)) * "\n"
        end
      else
        s3p = "$(tmp_decls)" * "\n" *
              "$(y[1])_r, $(y[1])_i" * foldl(*, vmap(i -> ", $(y[i])_r, $(y[i])_i", 2:n2)) *
              " = " *
              (w[1] == "1" ? "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" : "$(sat_expr("tmp$(t[1])", "$(w[1])"))") *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-2)", "$(w[i])"))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i" * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(w[n2+1])"))" *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", "$(w[n2+i])"))", 2:n2)) * "\n"
    end
  end
  end
  s = n == MODULO ? load_real_imag_gen(x; mode=mode, T=T) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
  return s
end


# Helper function to reinterpret complex array as float array
@inline function complex_to_float_zerocopy(input::Vector{ComplexF16})
    ptr = reinterpret(Ptr{Float16}, pointer(input))
    return unsafe_wrap(Vector{Float16}, ptr, 2*length(input), own=false)
end

@inline function complex_to_float_zerocopy(input::Vector{ComplexF32})
    ptr = reinterpret(Ptr{Float32}, pointer(input))
    return unsafe_wrap(Vector{Float32}, ptr, 2*length(input), own=false)
end

@inline function complex_to_float_zerocopy(input::Vector{ComplexF64})
    ptr = reinterpret(Ptr{Float64}, pointer(input))
    return unsafe_wrap(Vector{Float64}, ptr, 2*length(input), own=false)
end