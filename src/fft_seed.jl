load_reim = t -> join(["$(i == 1 ? "" : " ") $(t[i]) = reim(x[$(match(r"\d+$", t[i]).match)])" for i in 1:length(t)], ";")
#load_reim = t -> join(["$(i == 1 ? "" : " ") x$i = reim(x[$(match(r"\d+$", t[i]).match)])" for i in 1:length(t)], ";")
# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int,  suffixes::Vector{String}, D_status::Int, ::Type{T}) where T <: AbstractFloat

    global inc = inccounter() #nullify glabal tmp 't' var counter for each new kernel generated

    input = "y" ∈ suffixes ? "y" : "x"
    output = "y"
    d_matrix = "D"
    is_mat = "mat" ∈ suffixes
    
    if is_mat
        x = ["$input[k, $i]" for i in 1:n]
        y = ["$output[k, $i]" for i in 1:n]
        d = D_status == 2 ? ["$d_matrix[k, $i]" for i in 1:(n-1)] : D_status == 1 ? ["$d_matrix[$i]" for i in 1:(n-1)] : nothing
    else
        #x = ["$input[$i]" for i in 1:n]
        x = ["$(input)$i" for i in 1:n]
        #x = ["$(input)$(mod(i-1, 4) + 1)" for i in 1:n]
        y = ["$output[$i]" for i in 1:n]
        d = nothing
    end

    s = recfft2(y, x, d, nothing, true, T) # Replace with any other recfft kernel family seed.
    
    kernel_code = replace(s, 
            "#INPUT#" => input,
            "#OUTPUT#" => output)

    return kernel_code
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

function add_more_tmp_vars(x1, x2, wn, n)
    index = 0
    tmp_parts = String[]
    x_parts = String[]
    for w in wn
        if w ∉  ("-im", "1")
            for i in 1:n
                push!(tmp_parts, "tmp$(index)_r", "tmp$(index)_i")
                push!(x_parts, x1[i], x2[i])
                index += 1
            end
        end
    end
    tmp_vars = join(tmp_parts, ", ")
    x_vars = join(x_parts, ", ")
    return "$tmp_vars = $x_vars"
end

function sat_expr(tmp, w, d=nothing)
    
    if w == "-im"
        #is_t = startswith(x1, "t") || startswith(x2, "t")
        # -i*(a ± b) = ±(b_i ∓ a_i) ± i*(b_r ∓ a_r)
        #return is_t ? 
            #"$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" :
            return "$(tmp)_i, $(tmp)_r"
    
    elseif w == "INV_SQRT2_Q4"
        # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
        return "INV_SQRT2*($(tmp)_r + $(tmp)_i), " *
               "INV_SQRT2*($(tmp)_i - $(tmp)_r)"
    
    elseif w == "-INV_SQRT2_Q1"
        # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ∓ b_i)/√2 ]
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
        end
    end
end

function sat_expr(sign, x1, x2, w, d=nothing)
  @show w, x1, sign, x2
    
    if w == "-im"
        is_t = startswith(x1, "t") || startswith(x2, "t")
        # -i*(a ± b) = ±(b_i ∓ a_i) ± i*(b_r ∓ a_r)
        return is_t ? 
            "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" :
            "$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
    
    elseif w == "INV_SQRT2_Q4"
        # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
        return "INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i)), " *
               "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r))"
    
    elseif w == "-INV_SQRT2_Q1"
        # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ∓ b_i)/√2 ]
        return "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r)), " *
               "-INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i))"
    
    else
        num, den, is_q1 = parse_cispi(w)
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"
        
        if startswith(w, "CISPI")
            return is_q1 ?
                # Q1: cosθ + i sinθ
                "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
                "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))" :
                # Q4: cosθ - i sinθ
                "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
                "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))"
        
        elseif startswith(w, "-im*CISPI")
            return is_q1 ?
                # -i*(cosθ + i sinθ) = sinθ - i cosθ
                "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
                "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))" :
                # -i*(cosθ - i sinθ) = -sinθ - i cosθ
                "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
                "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))"
        
        elseif startswith(w, "-CISPI")
            return is_q1 ?
                # -cosθ - i sinθ
                "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
                "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))" :
                # -cosθ + i sinθ
                "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
                "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))"
        end
    end
end

function recfft2(y, x, d, w, root, ::Type{T}) where T <: AbstractFloat
  n = length(x)
  use_vars = false

  if n == 1
    ""
  elseif n == 2
    if root
    s = load_reim(x) * "\n" * """
        $(y[1]), $(y[2]) = Complex{$T}($(x[1])[1] + $(x[2])[1], $(x[1])[2] + $(x[2])[2]), Complex{$T}($(x[1])[1] - $(x[2])[1], $(x[1])[2] - $(x[2])[2])
        """
    else
    s = if !isnothing(d)
      if isnothing(w)
        """
        $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(d[1])*($(x[1]) - $(x[2]))
        """
      else
        w[1] == "1" ? 
        """
        $(y[1]), $(y[2]) = ($(x[1]) + $(x[2])), ($(d[1])*$(w[2]))*($(x[1]) - $(x[2]))
        """ : 
        """
        $(y[1]), $(y[2]) = ($(w[1]))*($(x[1]) + $(x[2])), ($(d[1])*$(w[2]))*($(x[1]) - $(x[2]))
        """
      end
    else
      if isnothing(w)
        """
        $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])[1] + $(x[2])[1], $(x[1])[2] + $(x[2])[2], $(x[1])[1] - $(x[2])[1], $(x[1])[2] - $(x[2])[2]
        """
      else
        w[1] == "1" ? 
        """
        $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(x[1])[1] + $(x[2])[1], $(x[1])[2] + $(x[2])[2], $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
        """ :
        """
        $(y[1]), $(y[2]) = ($(w[1]))*($(x[1]) + $(x[2])), ($(w[2]))*($(x[1]) - $(x[2]))
        """
      end
    end
    end
    #s = root ? load_reim(n) * "\n" * s : s
    return s
  else
    t = vmap(i -> "t$(inc())", 1:n)
    n2 = n ÷ 2
    wn = get_twiddle_expression(collect(0:n2-1), n)
    
    # Recursively handle sub-transforms
    s1 = recfft2(t[1:n2], x[1:2:n], nothing, nothing, false, T)
    s2 = recfft2(t[n2+1:n], x[2:2:n], nothing, wn, false, T)
    tmp = ""
    
    # Final layer combining with D matrix twiddles
    if !isnothing(d)
      if isnothing(w)
        s3p = "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:n2)) *
              " = " *
              "$(t[1]) + $(t[1+n2])" * foldl(*, vmap(i -> ",$(d[i-1])*($(t[i]) + $(t[i+n2]))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ",$(y[i+n2])", 2:n2)) *
              " = " *
              "$(d[n2])*($(t[1]) - $(t[1+n2]))" * foldl(*, vmap(i -> ",$(d[i+n2-1])*($(t[i]) - $(t[i+n2]))", 2:n2)) * "\n"
      else
        s3p = "$(y[1])" * foldl(*, vmap(i -> ", $(y[i])", 2:n2)) *
              " = " *
              (w[1] == "1" ? "$(t[1]) + $(t[1+n2])" : "($(w[1]))*($(t[1]) + $(t[1+n2]))") *
              foldl(*, vmap(i -> ", $(d[i-1])*($(w[i]))*($(t[i]) + $(t[i+n2]))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ", $(y[i+n2])", 2:n2)) *
              " = " *
              "($(d[n2])*$(w[n2+1]))*($(t[1]) - $(t[1+n2]))" *
              foldl(*, vmap(i -> ", ($(d[i+n2-1])*$(w[n2+i]))*($(t[i]) - $(t[i+n2]))", 2:n2)) * "\n"
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
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i " * foldl(*, vmap(i -> ",$(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i", 2:n2)) * "\n"
        end
      else
        if use_vars
        s3p = add_more_tmp_vars(["$(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i" for i in 2:n2], ["$(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i" for i in 2:n2], ["$(w[i])" for i in 2:n2], n2-1) * "\n" *
              "$(y[1])_r, $(y[1])_i" * foldl(*, vmap(i -> ", $(y[i])_r, $(y[i])_i", 2:n2)) *
              " = " *
              (w[1] == "1" ? "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" : "$(sat_expr("tmp$(t[1])", "$(w[1])"))") *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-2)", "$(w[i])"))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i" * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(w[n2+1])"))" *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", "$(w[n2+i])"))", 2:n2)) * "\n"
        else
        s3p = "$(y[1])_r, $(y[1])_i" * foldl(*, vmap(i -> ", $(y[i])_r, $(y[i])_i", 2:n2)) *
              " = " *
              (w[1] == "1" ? "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" : "$(sat_expr("+", "$(t[1])", "$(t[1+n2])", "$(w[1])"))") *
              foldl(*, vmap(i -> ", $(sat_expr("+", "$(t[i])", "$(t[i+n2])", "$(w[i])"))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i" * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(w[n2+1])"))" *
              foldl(*, vmap(i -> ", $(sat_expr("-", "$(t[i])", "$(t[i+n2])", "$(w[n2+i])"))", 2:n2)) * "\n"
        end
    end
    s = n == 4 ? load_reim(x) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
    return s
  end
  end
end
