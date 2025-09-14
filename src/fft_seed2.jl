# Modified functions for register-efficient FFT generation

# Helper function to generate let block variable unpacking
function generate_stage_inputs(x_vars, stage_name)
    input_assignments = String[]
    for x in x_vars
        m = match(r"(\d+)", x)
        if m !== nothing
            num = m.captures[1]
            var = startswith(x, "x") ? "x" :
                  startswith(x, "y") ? "y" :
                  startswith(x, "t") ? "t" :
                  startswith(x, "d") ? "d" : error("Unknown input: $x")
            
            if occursin('[', x)
                # Handle array access like x[5]
                rhs = replace(x, " " => "")
                push!(input_assignments, "$(var)$(num)_r, $(var)$(num)_i = real($rhs), imag($rhs)")
            else
                # Handle simple variable like x5
                push!(input_assignments, "$(var)$(num)_r, $(var)$(num)_i = real($(x)), imag($(x))")
            end
        end
    end
    return isempty(input_assignments) ? "" : join(input_assignments, "\n            ")
end

# Generate stage output tuple for register reuse
function generate_stage_output(results, stage_name)
    if isempty(results)
        return "()"
    end
    
    output_exprs = String[]
    for result in results
        push!(output_exprs, result)
    end
    
    return "(\n                " * join(output_exprs, ",\n                ") * "\n            )"
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
      #return "local $(join(tmp_vars, ", ")) = $(join(assignments, ", "))\n"
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
      #return is_t ? 
          "$(x1)_r $sign $(x2)_r, $(x1)_i $sign $(x2)_i" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
  elseif w == "-im"
      # -i*(a ± b) = ±(b_i ∓ a_i) ± i*(b_r ∓ a_r)
      #return is_t ? 
          "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
      #return is_t ? 
          "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
  elseif w == "INV_SQRT2_Q4"
      # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
      #return is_t ?
      #return is_t ?
          "INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i)), " *
          "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r))" #:
          #"INV_SQRT2*(($(x1)[1] $sign $(x2)[1]) + ($(x1)[2] $sign $(x2)[2])), " *
          #"INV_SQRT2*(($(x1)[2] $sign $(x2)[2]) - ($(x1)[1] $sign $(x2)[1]))"
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

# Modified recursive FFT function with register optimization
function recfft2(y, x, d, w, root, ::Type{T}, stage_counter=1, path="") where T <: AbstractFloat
    n = length(x)
    
    if n == 1
        return ""
    elseif n == 2
        stage_name = "stage$(stage_counter)"
        
        #=
        if !isnothing(d) && !isnothing(w)
            error("Cannot have both d and w non-nothing")
        end
        =#
        
        if root
            input_load = generate_stage_inputs(x, stage_name)
            if !isnothing(d)
                return """
        $stage_name = let
            $input_load
            tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            (
                $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i,  # y[1] components
                $(sat_expr("tmp0", "$(d[1])"))                    # y[2] components  
            )
        end
        
        $(y[1]), $(y[2]) = Complex{$T}($stage_name[1], $stage_name[2]), Complex{$T}($stage_name[3], $stage_name[4])
        """
            else
                return """
        $stage_name = let
            $input_load
            (
                $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i,  # y[1] components
                $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i   # y[2] components
            )
        end
        
        $(y[1]), $(y[2]) = Complex{$T}($stage_name[1], $stage_name[2]), Complex{$T}($stage_name[3], $stage_name[4])
        """
            end
        else
            # Non-root case - return tuple components for next stage
            if !isnothing(w)
                if w[1] == "1"
                    return """
            ($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])")))"""
                else
                    return """
            ($(sat_expr("+", "$(x[1])", "$(x[2])", "$(w[1])")), $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])")))"""
                end
            else
                return """
            ($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i)"""
            end
        end
    else
        n2 = n ÷ 2
        t = ["t$i" for i in stage_counter*100:(stage_counter*100 + n - 1)]  # Unique temps per stage
        
        # Generate left and right recursive calls
        left_stage = stage_counter * 10 + 1
        right_stage = stage_counter * 10 + 2
        final_stage = stage_counter
        
        s1 = recfft2(t[1:n2], x[1:2:n], nothing, nothing, false, T, left_stage, "$(path)L")
        s2 = recfft2(t[n2+1:n], x[2:2:n], nothing, get_twiddle_expression(collect(0:n2-1), n), false, T, right_stage, "$(path)R")
        
        # Generate the combining stage
        stage_name = "stage$(final_stage)"
        input_load = generate_stage_inputs(x, stage_name)
        
        # Generate butterfly operations within let block
        butterfly_ops = String[]
        
        if root
            # Root case - generate final outputs
            if !isnothing(d)
                # With D matrix
                push!(butterfly_ops, "# Left recursive stage")
                push!(butterfly_ops, s1)
                push!(butterfly_ops, "# Right recursive stage") 
                push!(butterfly_ops, s2)
                
                # Combine results
                output_assignments = String[]
                for i in 1:n2
                    if i == 1
                        push!(output_assignments, "$(y[i]) = Complex{$T}($(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i)")
                    else
                        push!(output_assignments, "$(y[i]) = Complex{$T}($(sat_expr("$(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i", "$(d[i])")))")
                    end
                end
                
                for i in 1:n2
                    if i == 1
                        push!(output_assignments, "$(y[i+n2]) = Complex{$T}($(sat_expr("-", "$(t[i])", "$(t[i+n2])", "$(d[i+n2])")))")
                    else
                        push!(output_assignments, "$(y[i+n2]) = Complex{$T}($(sat_expr("-", "$(t[i])", "$(t[i+n2])", "$(d[i+n2])")))")
                    end
                end
                
                return """
        $stage_name = let
            $input_load
            
            # Recursive stages with register reuse
            left_results = let
                $s1
            end
            
            right_results = let  
                $s2
            end
            
            # Final butterfly combining
            $(join(output_assignments, "\n            "))
        end
        """
            else
                # Without D matrix
                output_assignments = String[]
                for i in 1:n2
                    push!(output_assignments, "$(y[i]) = Complex{$T}($(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i)")
                    push!(output_assignments, "$(y[i+n2]) = Complex{$T}($(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i)")
                end
                
                return """
        $stage_name = let
            $input_load
            
            # Recursive stages with register reuse
            left_results = let
                $s1
            end
            
            right_results = let
                $s2  
            end
            
            # Final butterfly combining
            $(join(output_assignments, "\n            "))
        end
        """
            end
        else
            # Non-root case - return components for next stage
            result_components = String[]
            
            if !isnothing(w)
                # With twiddle factors
                for i in 1:n2
                    if w[i] == "1"
                        push!(result_components, "$(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i")
                    else
                        push!(result_components, "$(sat_expr("+", "$(t[i])", "$(t[i+n2])", "$(w[i])"))")
                    end
                end
                
                for i in 1:n2
                    push!(result_components, "$(sat_expr("-", "$(t[i])", "$(t[i+n2])", "$(w[i+n2])"))")
                end
            else
                # Without twiddle factors
                for i in 1:n2
                    push!(result_components, "$(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i")
                    push!(result_components, "$(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i")
                end
            end
            
            return """
        let
            $input_load
            
            # Left stage
            left_stage = let
                $s1
            end
            
            # Right stage  
            right_stage = let
                $s2
            end
            
            # Return combined results
            $(generate_stage_output(result_components, stage_name))
        end
        """
        end
    end
end

function inccounter()
  let counter = 0
    return () -> (counter += 1)
  end
end

inc = inccounter()

# Modified main function to use optimized version
function makefftradix(n::Int, suffixes::SuffixFlags, D::AbstractArray{String}, p::Int, s::Int, SIZE::Int, ::Type{T}) where T <: AbstractFloat
    global inc = inccounter() # Reset counter for each new kernel
    
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
    
    # Generate optimized kernel code
    kernel_code = recfft2(y, x, d, nothing, true, T, 1, "") |> 
                  s -> replace(s, "#INPUT#" => input, "#OUTPUT#" => output) #|> 
                  #s -> replace_with_reused_vars(s)
    
    # Parse and return
    if isempty(kernel_code)
        return quote end
    else
        try
            # Wrap in @inbounds @fastmath for performance
            wrapped_code = """
            @inbounds @fastmath begin
                $kernel_code
            end
            """
            parsed_expr = Meta.parse(wrapped_code)
            return parsed_expr
        catch e
            @warn "Failed to parse optimized kernel code: $e"
            @warn "Kernel code was: $kernel_code"
            return quote
                copyto!(y, x)
            end
        end
    end
end