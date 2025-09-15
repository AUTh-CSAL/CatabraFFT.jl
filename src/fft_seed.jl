include("suffix.jl")

using SIMD

"""
# Usage examples:
load_real_imag_gen(["x1", "x2"], mode=:default)
load_real_imag_gen(["x1", "x2"], mode=:unsafe_load, ptr_name="data_ptr")
load_real_imag_gen(["x1", "x2"], mode=:vload_soa, vec_width=8)
"""
load_real_imag_gen = (t; mode, ptr_name="px", vec_width=4) -> begin
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
    elseif mode == :vload_soa
        # Actual SIMD AVX2 / NEON (e.g. x86 ymm registers) loads for Structure of Arrays
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                "$(prefix)$(var)$(num)_vec = vload(Vec{$vec_width,ComplexF32}, $ptr_name, $(num-1)*$vec_width + 1)"
            end
            for (i, s) in enumerate(t)
        ], "; ")
    elseif mode == :vload_aos
        # SIMD loads for Array of Structures (interleaved real/imag)
        join([
            let
                m = match(r"(\d+)\D*$", s)
                num = parse(Int, m.captures[1])
                var = startswith(s, "x") ? "x" :
                      startswith(s, "y") ? "y" :
                      startswith(s, "D") ? "d" : error("Unknown input: $s")
                prefix = i == 1 ? "" : " "
                offset = (num-1)*2*vec_width
                "$(prefix)$(var)$(num)_ri = vload(Vec{$(2*vec_width),Float32}, $ptr_name, $offset + 1)"
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
  
  mode = :default

  has_y = has_flag(suffixes, Y)
  has_mat = has_flag(suffixes, MAT)
  has_vec = has_flag(suffixes, VEC)
  #@show has_vec has_mat has_y
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
  px = (mode != :default) ? "p$(input) = pointer(reinterpret($T, $(input)));" : ""
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
                load_real_imag_gen(x; mode=mode) * "\n" * 
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" * """
                $(y[1]), $(y[2]), $(y[3]), $(y[4]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(sat_expr("tmp0", "$(d[1])"))
                """
              else
                load_real_imag_gen(x; mode=mode) * "\n" * 
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" * """
                $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(sat_expr("tmp0", "$(d[1])")))
                """
              end
            end
          end
        else
          if root
            if mode == :unsafe_load
            load_real_imag_gen(x; mode=mode) * "\n" * "$py" * "\n" * """
            $(y[1]), $(y[2]), $(y[3]), $(y[4]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            """ 
            else
            load_real_imag_gen(x; mode=mode) * "\n" * """
            $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i)
            """ 
            end #TODO CONTINUE FROM HERE
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
  s = n == MODULO ? load_real_imag_gen(x; mode=mode) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
  return s
end