
load_reim = n -> join(["$(i == 1 ? "" : " ") x$i = reim(x[$i])" for i in 1:n], ";")

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
        x = ["$input[$i]" for i in 1:n]
        #x = ["$(input)$i" for i in 1:n]
        y = ["$output[$i]" for i in 1:n]
        d = nothing
    end

    s = recfft2(y, x, d, nothing, true) # Replace with any other recfft kernel family seed.
    
    kernel_code = replace(s, 
            "#INPUT#" => input,
            "#OUTPUT#" => output)

    return kernel_code
end

# FFT8 optimized with SIMD and reduced temporaries
@inline function fft8_shell!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64})
    inv_sqrt2 = 1 /sqrt(2)
    @inbounds begin
        # Load all elements first using SIMD-friendly pattern
        x1 = reim(x[1])
        x2 = reim(x[2])
        x3 = reim(x[3])
        x4 = reim(x[4])
        x5 = reim(x[5])
        x6 = reim(x[6])
        x7 = reim(x[7])
        x8 = reim(x[8])
        
        # First layer butterflies
        t9_r = x1[1] + x5[1]; t9_i = x1[2] + x5[2]
        t10_r = x1[1] - x5[1]; t10_i = x1[2] - x5[2]
        
        t11_r = x3[1] + x7[1]; t11_i = x3[2] + x7[2]
        t12_r = x3[2] - x7[2]; t12_i = x7[1] - x3[1]  # -im*(x3 - x7)
        
        t1_r = t9_r + t11_r; t1_i = t9_i + t11_i
        t2_r = t10_r + t12_r; t2_i = t10_i + t12_i
        t3_r = t9_r - t11_r; t3_i = t9_i - t11_i
        t4_r = t10_r - t12_r; t4_i = t10_i - t12_i
        
        # Second layer
        t13_r = x2[1] + x6[1]; t13_i = x2[2] + x6[2]
        t14_r = x2[1] - x6[1]; t14_i = x2[2] - x6[2]
        
        t15_r = x4[1] + x8[1]; t15_i = x4[2] + x8[2]
        t16_r = x4[2] - x8[2]; t16_i = x8[1] - x4[1]  # -im*(x4 - x8)
        
        # Twiddle factor multiplications using precomputed constants
        t5_r = t13_r + t15_r; t5_i = t13_i + t15_i
        t6_r = INV_SQRT2 * (t14_r + t16_r)
        t6_i = INV_SQRT2 * (t14_i + t16_i)
        
        t7_r = t13_i - t15_i; t7_i = t15_r - t13_r  # -im*(t13 - t15)
        t8_r = -INV_SQRT2 * (t14_r - t16_r)
        t8_i = -INV_SQRT2 * (t14_i - t16_i)
        
        # Final combinations
        y[1] = ComplexF64(t1_r + t5_r, t1_i + t5_i)
        y[2] = ComplexF64(t2_r + t6_r, t2_i + t6_i)
        y[3] = ComplexF64(t3_r + t7_r, t3_i + t7_i)
        y[4] = ComplexF64(t4_r + t8_r, t4_i + t8_i)
        y[5] = ComplexF64(t1_r - t5_r, t1_i - t5_i)
        y[6] = ComplexF64(t2_r - t6_r, t2_i - t6_i)
        y[7] = ComplexF64(t3_r - t7_r, t3_i - t7_i)
        y[8] = ComplexF64(t4_r - t8_r, t4_i - t8_i)
    end
    return y
end

function sat_expr(sign, x1, x2, w, d=nothing) 
    
end

function recfft2(y, x, d=nothing, w=nothing, root=false)
  n = length(x)
  if n == 1
    ""
  elseif n == 2
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
        $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(x[1]) - $(x[2])
        """
      else
        w[1] == "1" ? 
        """
        $(y[1]), $(y[2]) = ($(x[1]) + $(x[2])), ($(w[2]))*($(x[1]) - $(x[2]))
        """ :
        """
        $(y[1]), $(y[2]) = ($(w[1]))*($(x[1]) + $(x[2])), ($(w[2]))*($(x[1]) - $(x[2]))
        """
      end
    end
    s = root ? load_reim(n) * "\n" * s : s
    return s
  else
    t = vmap(i -> "t$(inc())", 1:n)
    t_r = t .* "_r"; t_l = t .* "_l"
    n2 = n ÷ 2
    wn = get_twiddle_expression(collect(0:n2-1), n)
    
    # Recursively handle sub-transforms
    s1 = recfft2(t[1:n2], x[1:2:n], nothing, nothing)
    s2 = recfft2(t[n2+1:n], x[2:2:n], nothing, wn)
    
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
        s3p = "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:n2)) *
              " = " *
              "$(t[1]) + $(t[1+n2])" * foldl(*, vmap(i -> ",$(t[i]) + $(t[i+n2])", 2:n2)) * "\n"
        s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ",$(y[i+n2])", 2:n2)) *
              " = " *
              "$(t[1]) - $(t[1+n2])" * foldl(*, vmap(i -> ",$(t[i]) - $(t[i+n2])", 2:n2)) * "\n"
      else
        s3p = "$(y[1])" * foldl(*, vmap(i -> ", $(y[i])", 2:n2)) *
              " = " *
              (w[1] == "1" ? "$(t[1]) + $(t[1+n2])" : "($(w[1]))*($(t[1]) + $(t[1+n2]))") *
              foldl(*, vmap(i -> ", ($(w[i]))*($(t[i]) + $(t[i+n2]))", 2:n2)) * "\n"
        s3m = "$(y[n2+1])" * foldl(*, vmap(i -> ", $(y[i+n2])", 2:n2)) *
              " = " *
              "($(w[n2+1]))*($(t[1]) - $(t[1+n2]))" *
              foldl(*, vmap(i -> ", ($(w[n2+i]))*($(t[i]) - $(t[i+n2]))", 2:n2)) * "\n"
    end
    #s = root ? load_reim(n) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
    return s1 * s2 * s3p * s3m 
  end
  end
end
