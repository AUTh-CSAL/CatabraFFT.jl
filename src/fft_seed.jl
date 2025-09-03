include("suffix.jl")

load_real_imag = t -> join([
    let
        m = match(r"(\d+)\D*$", s)
        num = m.captures[1]
        var = startswith(s, "x") ? "x" :
              startswith(s, "y") ? "y" :
              startswith(s, "D") ? "d" : error("Unknown input: $s")
        rhs = occursin('[', s) ? replace(s, " " => "") : "$var[$num]"
        prefix = i == 1 ? "" : " "
        "local $(prefix)$(var)$(num)_r , $(prefix)$(var)$(num)_i = real($rhs), imag($rhs)"
    end
    for (i, s) in enumerate(t)
], "; ")

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

  has_y = has_flag(suffixes, Y)
  has_mat = has_flag(suffixes, MAT)
  has_vec = has_flag(suffixes, VEC)
  input = has_y ? "y" : "x"
  output = "y"
  groups = SIZE ÷ s
  
  if has_mat
      if has_vec
        x = ["$(input)$(i + p*s)" for i in 1:n]
        y = ["$output[$(i + p*s)]" for i in 1:n]
      else
        x = ["$(input)$(p + 1 + (i-1)*groups)" for i in 1:n]
        y = ["$output[$(i + p*s)]" for i in 1:n]
      end
      d = D == String[] ? nothing : D
  else
    if has_vec
      x = ["$(input)($i + offset)" for i in 1:n]
      y = ["$output[$i + offset]" for i in 1:n]
    else
      x = ["$(input)$i" for i in 1:s:n]
      y = ["$output[$i]" for i in 1:n]
    end
    d = nothing
  end

  # Generate kernel code as string first
  kernel_code = recfft2(y, x, d, nothing, true, T) |> s -> replace(s, "#INPUT#" => input, "#OUTPUT#" => output) |> s -> replace_with_reused_vars(s)
  
  # Parse the string into actual Julia expressions
  if isempty(kernel_code)
      return quote end
  else
      try
          # Wrap in begin...end block for parsing multiple statements
          parsed_expr = Meta.parse("begin\n$kernel_code\nend")
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
      return "local $(join(tmp_vars, ", ")) = $(join(assignments, ", "))\n"
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

function recfft2(y, x, d, w, root, ::Type{T}, tmp_base=1) where T <: AbstractFloat
  n = length(x)
  MODULO = 4

  if n == 1
    ""
  elseif n == 2
    s = if !isnothing(d)
          if isnothing(w)
            if root
              load_real_imag(x) * "\n" * 
              "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * """
              $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(sat_expr("tmp0", "$(d[1])")))
              """
            end
          end
        else
          if root
            load_real_imag(x) * "\n" * """
            $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i)
            """ 
          else
            if isnothing(w)
            """
            local $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            """
            else
            w[1] == "1" ? 
                """
                local $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
                """ :
                """
                local $(y[1]), $(y[2]) = $(sat_expr("+", "$(x[1])", "$(x[2])", "$(w[1])")), $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
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
    s1 = recfft2(t[1:n2], x[1:2:n], nothing, nothing, false, T, new_tmp_base)
    s2 = recfft2(t[n2+1:n], x[2:2:n], nothing, get_twiddle_expression(collect(0:n2-1), n), false, T, new_tmp_base)

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
        s3p = "$(tmp_decls)" * "\n" *
              "$(y[1])" * foldl(*, vmap(i -> ", $(y[i])", 2:n2)) *
              " = " *
              "Complex{$T}($(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(sat_expr("tmp$(i-2)", "$(d[i-1])")))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ", $(y[i+n2])", 2:n2)) *
              " = " *
              "Complex{$T}($(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(d[n2])")))" * foldl(*, vmap(i -> ", Complex{$T}($(sat_expr("tmp$(i-3+n2)", "$(d[i+n2-1])")))", 2:n2)) * "\n"
        end
      end
    else
      if isnothing(w)
        if root 
          s3p = "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:n2)) *
                " = " *
                "Complex{$T}($(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i)", 2:n2)) * "\n"
          s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ",$(y[i+n2])", 2:n2)) *
                " = " *
                "Complex{$T}($(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i)" * foldl(*, vmap(i -> ", Complex{$T}($(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i)", 2:n2)) * "\n"
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
        s3m = "local $(y[n2+1])_r, $(y[n2+1])_i" * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(w[n2+1])"))" *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", "$(w[n2+i])"))", 2:n2)) * "\n"
    end
  end
  end
  s = n == MODULO ? load_real_imag(x) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
  return s
end