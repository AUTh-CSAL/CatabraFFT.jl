include("suffix.jl")
using BenchmarkTools, SIMD

const SIMD_BITS = 256

"""
# Usage examples:
load_real_imag_gen(["x1", "x2"], mode=:default)
load_real_imag_gen(["x1", "x2"], mode=:unsafe_load, ptr_name="data_ptr")
load_real_imag_gen(["x1", "x2"], mode=:vload_soa, vec_width=8)
"""
load_gen = (t; mode, T, ptr_name="px", SIMD_BITS) -> begin
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
        # Smart detection: use vload for contiguous, vgather for strided
        complex_size_bits = 2 * sizeof(T) * 8
        complexes_per_vec = SIMD_BITS ÷ complex_size_bits
        n_floats = 2*length(t)
        
        # Extract indices to check if contiguous
        indices = Int[]
        for s in t
            m = match(r"(\d+)\D*$", s)
            push!(indices, parse(Int, m.captures[1]))
        end
        
        # Check if indices are contiguous
        is_contiguous = (length(indices) > 1) && all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))

        # Check for pairwise contiguous pattern (e.g., [1,2,9,10] or [5,6,13,14])
        is_pairwise_contiguous = (length(indices) % 2 == 0) && all(i -> indices[2i] == indices[2i-1] + 1, 1:(length(indices)÷2))

        if is_contiguous && n_floats <= complexes_per_vec
            # Use vload for contiguous access - much faster!
            code_parts = String[]
            
            # For vload, we need a pointer, not reinterpret
            push!(code_parts, "$(ptr_name)_ptr = pointer(reinterpret($T, $ptr_name))")
            
            # Calculate offset for first element
            offset = 2 * (indices[1] - 1)  # Convert to float index (0-based)
            
            # Load contiguous data with single instruction
            if n_floats == complexes_per_vec
              push!(code_parts, "v1, v2 = vload(Vec{$(n_floats÷2), $T}, $(ptr_name)_ptr + $offset), vload(Vec{$(n_floats÷2), $T}, $(ptr_name)_ptr + $(offset + n_floats÷2*sizeof(T)))")
            else
              push!(code_parts, "v = vload(Vec{$(n_floats),$T}, $(ptr_name)_ptr + $offset)")
            end
            join(code_parts, "\n    ")
            
          elseif !is_contiguous && n_floats <= complexes_per_vec
            # Non-contiguous: use vgather
            code_parts = String[]
            
            if n_floats <= complexes_per_vec

              offset = 2 * (indices[1] - 1)  # Convert to float index (0-based)

              push!(code_parts, "v1 = vload(Vec{$(n_floats÷2), $T}, $(ptr_name) + $(offset));")
              push!(code_parts, "v2 = vload(Vec{$(n_floats÷2), $T}), $(ptr_name) + $(offset + n_floats÷2*sizeof(T)));")
            
              #=
            elseif 
                # Single vgather
                float_indices = Int[]
                for num in indices
                    push!(float_indices, 2*num - 1)  # real index
                    push!(float_indices, 2*num)      # imag index
                end
                
                idx_tuple = "(" * join(float_indices, ",") * ")"
                #push!(code_parts, "$(ptr_name)= complex_to_float_zerocopy($ptr_name)")
                push!(code_parts, "idx = Vec{$(n_floats),Int64}($idx_tuple)")
                push!(code_parts, "v = vgather($(ptr_name), idx)")
                
            =#
            else
                # Multiple vgathers for large radix
                for chunk_start in 1:complexes_per_vec:n_floats
                    chunk_end = min(chunk_start + complexes_per_vec - 1, n_floats)
                    chunk_indices = indices[chunk_start:chunk_end]
                    
                    float_indices = Int[]
                    for num in chunk_indices
                        push!(float_indices, 2*num - 1)
                        push!(float_indices, 2*num)
                    end
                    
                    while length(float_indices) < 2*complexes_per_vec
                        push!(float_indices, 1)
                    end
                    
                    chunk_id = (chunk_start - 1) ÷ complexes_per_vec + 1
                    idx_tuple = "(" * join(float_indices[1:2*complexes_per_vec], ",") * ")"
                    
                    push!(code_parts, "idx$(chunk_id) = Vec{$(2*complexes_per_vec),Int64}($idx_tuple)")
                    push!(code_parts, "v$(chunk_id) = vgather($(ptr_name)_floats, idx$(chunk_id))")
                    
                end
            end
            
            join(code_parts, "\n    ")
        end
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

store_gen = (y, t_vars; mode, T, ptr_name="py", SIMD_BITS) -> begin
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complexes_per_vec
    n_floats = 2*length(indices)
    @show complex_size_bits, complexes_per_vec, n_floats

    # Extract indices from y array pattern
    indices = Int[]
    for yi in y
        m = match(r"\[(\d+)\]", yi)
        if m !== nothing
            push!(indices, parse(Int, m.captures[1]))
        end
    end
    # TODO add twiddle shuffle saturation arithmetic operations here.
    
    # Check if indices are contiguous
    is_contiguous = (length(indices) > 1) && all(i ->indices[i] == indices[1] + i - 1, 2:length(indices))

    code_parts = String[]
    
    # Use vstore for contiguous access - much faster!
    if is_contiguous && n_floats <= complexes_per_vec
        
        # For vstore, we need a pointer
        push!(code_parts, "$(ptr_name)_ptr = pointer(reinterpret($T, $ptr_name))")
        
        # Calculate offset for first element
        offset = 2 * (indices[1] - 1)  # Convert to float index (0-based)
        
        # Create vector from t values
        vals = String[]
        for (i, t_var) in enumerate(t_vars)
            push!(vals, "$(t_var)_r")
            push!(vals, "$(t_var)_i")
        end
        vals_tuple = "(" * join(vals, ",") * ")"

        push!(code_parts, "v = Vec{$(n_floats),$T}($vals_tuple)")
        push!(code_parts, "vstore(v, $(ptr_name)_ptr + $offset)")
        
        join(code_parts, "\n    ")
        
    elseif !is_contiguous && n_floats <= complexes_per_vec
        # Non-contiguous: use vscatter
        
        # Single vscatter for small radix
        float_indices = Int[]
        for idx in indices
            push!(float_indices, 2*idx - 1)  # real index
            push!(float_indices, 2*idx)      # imag index
        end
        
        # Create values vector from t variables
        vals = String[]
        for (i, t_var) in enumerate(t_vars)
            push!(vals, "$(t_var)_r")
            push!(vals, "$(t_var)_i")
        end
        
        idx_tuple = "(" * join(float_indices, ",") * ")"
        vals_tuple = "(" * join(vals, ",") * ")"
        @show idx_tuple, vals_tuple
        
        push!(code_parts, "idx = Vec{$(n_floats), Int64}($idx_tuple)")
        push!(code_parts, "v = Vec{$(n_floats),$T}($vals_tuple)")
        push!(code_parts, "vscatter(out_vec, $(ptr_name), idx)")
        
        join(code_parts, "\n    ")
        
    else
        # Multiple vscatters for large radix
        for chunk_start in 1:complexes_per_vec:n_floats
            chunk_end = min(chunk_start + complexes_per_vec - 1, n_floats)
            chunk_indices = indices[chunk_start:chunk_end]
            chunk_t_vars = t_vars[chunk_start:chunk_end]
            
            float_indices = Int[]
            for idx in chunk_indices
                push!(float_indices, 2*idx - 1)
                push!(float_indices, 2*idx)
            end
            
            vals = String[]
            for t_var in chunk_t_vars
                push!(vals, "$(t_var)_r")
                push!(vals, "$(t_var)_i")
            end
            
            # Pad if needed
            while length(float_indices) < 2*complexes_per_vec
                push!(float_indices, 1)
                push!(vals, "0")
            end
            
            chunk_id = (chunk_start - 1) ÷ complexes_per_vec + 1
            idx_tuple = "(" * join(float_indices[1:2*complexes_per_vec], ",") * ")"
            vals_tuple = "(" * join(vals[1:2*complexes_per_vec], ",") * ")"
            @show idx_tuple, vals_tuple
            
            push!(code_parts, "out_idx$(chunk_id) = Vec{$(2*complexes_per_vec),Int64}($idx_tuple)")
            push!(code_parts, "out_vec$(chunk_id) = Vec{$(2*complexes_per_vec),$T}($vals_tuple)")
            push!(code_parts, "vscatter(out_vec$(chunk_id), $(ptr_name), out_idx$(chunk_id))")
        end
        
        join(code_parts, "\n    ")
    end
end

# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int,  suffixes::SuffixFlags, D::AbstractArray{String}, p::Int, s::Int, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat

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

  px = if (mode == :vgather) "p$(input) = pointer(reinterpret($T, $(input)));"
      else "" end
  py = (mode == :unsafe_load) ? "$(output) = reinterpret($T, $(prev_output));" : ""
  kernel_code = recfft2_simd(y, x, d, nothing, true, T, 1, mode, py, SIMD_BITS) 
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

"""
Apply twiddle factor w to vector containing n_complex complex numbers
"""
function sat_expr_vec(vec_name, w, T, n_complex)
    if w == "1"
        # Identity - no operation needed
        return vec_name
        
    elseif w == "-im"
        # -im rotation: swap real/imag and negate imaginary
        # For [r1,i1,r2,i2,...] -> [i1,-r1,i2,-r2,...]
        swap_indices = Int[]
        for i in 1:n_complex
            push!(swap_indices, 2*i)     # imaginary first
            push!(swap_indices, 2*i - 1) # real second
        end
        
        neg_pattern = join([i % 2 == 0 ? "-1" : "1" for i in 1:2*n_complex], ",")
        
        return """shufflevector($vec_name, Val(($(join(swap_indices.-1, ","))))) * Vec{$(2*n_complex),$T}(($neg_pattern))"""
        
    elseif w == "INV_SQRT2_Q4"
        # (1-i)/√2 transformation
        # (a+bi) * (1-i)/√2 = [(a+b)/√2, (b-a)/√2]
        
        # First create sum and diff vectors
        sum_indices = []
        diff_indices = []
        for i in 1:n_complex
            r_idx = 2*i - 1
            i_idx = 2*i
            push!(sum_indices, "$vec_name[$r_idx] + $vec_name[$i_idx]")
            push!(diff_indices, "$vec_name[$i_idx] - $vec_name[$r_idx]")
        end
        
        return """Vec{$(2*n_complex),$T}(($(join(vcat(sum_indices, diff_indices), ","))) * INV_SQRT2)"""
        
    elseif startswith(w, "CISPI")
        # General twiddle factor application
        parsed = parse_cispi(w)
        c = "COSPI_$(parsed.num)_$(parsed.den)"
        s = "SINPI_$(parsed.num)_$(parsed.den)"
        
        # Create cos and sin vectors
        cos_sin_pattern = join([i % 2 == 1 ? c : s for i in 1:2*n_complex], ",")
        
        if parsed.q1
            # cos + i*sin
            return """complex_multiply($vec_name, Vec{$(2*n_complex),$T}(($cos_sin_pattern)))"""
        else
            # cos - i*sin
            sin_neg_pattern = join([i % 2 == 1 ? c : "-$s" for i in 1:2*n_complex], ",")
            return """complex_multiply($vec_name, Vec{$(2*n_complex),$T}(($sin_neg_pattern)))"""
        end
    else
        # Fallback for unhandled cases
        return vec_name
    end
end

# Helper for complex multiplication of vectors
@inline function complex_multiply(v1::Vec{N,T}, v2::Vec{N,T}) where {N,T}
  @fastmath @inbounds begin
    # v1 = [r1,i1,r2,i2,...], v2 = [c1,s1,c2,s2,...]
    # Result: [r1*c1-i1*s1, r1*s1+i1*c1, ...]
    
    # Extract real and imaginary parts
    v1_r = shufflevector(v1, Val(tuple([2i-1 for i in 1:N÷2]...)))
    v1_i = shufflevector(v1, Val(tuple([2i for i in 1:N÷2]...)))
    v2_r = shufflevector(v2, Val(tuple([2i-1 for i in 1:N÷2]...)))
    v2_i = shufflevector(v2, Val(tuple([2i for i in 1:N÷2]...)))
    
    # Complex multiplication
    res_r = muladd(v1_r, v2_r, -v1_i * v2_i)
    res_i = muladd(v1_r, v2_i, v1_i * v2_r)
    
    # Interleave back
    return shufflevector(res_r, res_i, Val(tuple(vcat([[2i-1,2i+N÷2-1] for i in 1:N÷2]...)...)))
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

function recfft2_simd(y, x, d, w, root, ::Type{T}, tmp_base=1, mode=:default, py="", SIMD_BITS) where T <: AbstractFloat
    n = length(x)
    MODULO = 4
    
    if n == 1
        ""
    elseif n == 2
        s = if !isnothing(d)
                if isnothing(w) # ROOT FFT2 KENREL
                    load_gen(x; mode=mode, T=T, ptr_name="px", SIMD_BITS) * "\n" *
                    "sum, diff = v1 + v2, v1 - v2" * "\n" * "$py" * 
                    store_gen(y, d; root=root, T=T, ptr_name="py", SIMD_BITS) 
                end 
            else
            if root
                load_gen(x; mode=mode, T=T, ptr_name="px", SIMD_BITS) * "\n" *
                ""
                
            
                    

end

# Helper function to apply twiddle factors using SIMD
function apply_twiddle_simd(vec_name, twiddle, T, n_complex)
    if twiddle == "1"
        ""
    elseif twiddle == "-im"
        # -im rotation: swap and negate real part
        """
        $vec_name = shufflevector($vec_name, Val((1,0))) * Vec{2,$T}((-1,1))
        """
    elseif twiddle == "INV_SQRT2_Q4"
        # (1-i)/√2
        """
        v_tmp_sum = shufflevector($vec_name, Val((0,0))) + shufflevector($vec_name, Val((1,1)))
        v_tmp_diff = shufflevector($vec_name, Val((1,1))) - shufflevector($vec_name, Val((0,0)))
        $vec_name = shufflevector(v_tmp_sum, v_tmp_diff, Val((0,2))) * Vec{2,$T}((INV_SQRT2, INV_SQRT2))
        """
    else
        # General twiddle - extract cos/sin values
        if startswith(twiddle, "CISPI")
            parsed = parse_cispi(twiddle)
            c = "COSPI_$(parsed.num)_$(parsed.den)"
            s = parsed.q1 ? "SINPI_$(parsed.num)_$(parsed.den)" : "-SINPI_$(parsed.num)_$(parsed.den)"
            
            """
            # Complex multiply by ($c, $s)
            v_r = $vec_name[1] * $c - $vec_name[2] * $s
            v_i = $vec_name[1] * $s + $vec_name[2] * $c
            $vec_name = Vec{2,$T}((v_r, v_i))
            """
        else
            ""
        end
    end
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