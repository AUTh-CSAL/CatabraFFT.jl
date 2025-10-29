include("suffix.jl")
include("radix_plan.jl")
using SIMD

const SIMD_BITS = 256

"""
# Usage examples:
load_real_imag_gen(["x1", "x2"], mode=:default)
load_real_imag_gen(["x1", "x2"], mode=:unsafe_load, ptr_name="data_ptr")
load_real_imag_gen(["x1", "x2"], mode=:vload_soa, vec_width=8)
"""
# Pointer-based scalar loads (e.g. xmm SSE4 registers)
# For small kernels utilizing Instruction-Level Parallelism (ILP)
# Modern CPUs can execute multiple independent scalar operations simultaneously:
# Can execute in parallel on different execution ports
# If your CPU likely has 2-3 ADD units, these execute in parallel despite being "scalar".
# No SIMD Setup Overhead! Shuffling data into SIMD layout, permuting for butterfly patterns and extracting results
# ...can be MORE expensive than simple scalar ops!
load_real_imag_gen = (t; mode, T, ptr_name="px") -> begin
    vec_width = 2sizeof(T)  # Width in bytes for real+imag pair
    
    if mode == :vgather
        # Smart detection: use vload for contiguous, vgather for strided
        avx2_bits = 256
        complex_size_bits = 2 * sizeof(T) * 8
        complexes_per_vec = avx2_bits ÷ complex_size_bits
        n_elems = length(t)
        
        # Extract indices to check if contiguous
        indices = Int[]
        for s in t
            m = match(r"(\d+)\D*$", s)
            push!(indices, parse(Int, m.captures[1]))
        end
        
        # Check if indices are contiguous
        is_contiguous = (length(indices) > 1) && all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
        
        if is_contiguous && n_elems <= complexes_per_vec
            # Use vload for contiguous access - much faster!
            code_parts = String[]
            
            # For vload, we need a pointer, not reinterpret
            push!(code_parts, "$(ptr_name)_ptr = pointer(reinterpret($T, $ptr_name))")
            
            # Calculate offset for first element
            offset = 2 * (indices[1] - 1)  # Convert to float index (0-based)
            
            # Load contiguous data with single instruction
            push!(code_parts, "v = vload(Vec{$(2*n_elems),$T}, $(ptr_name)_ptr + $offset)")
            join(code_parts, "\n    ")
            
          elseif !is_contiguous && n_elems <= complexes_per_vec
            # Non-contiguous: use vgather
            code_parts = String[]
            
            if n_elems <= complexes_per_vec
                # Single vgather
                float_indices = Int[]
                for num in indices
                    push!(float_indices, 2*num - 1)  # real index
                    push!(float_indices, 2*num)      # imag index
                end
                
                idx_tuple = "(" * join(float_indices, ",") * ")"
                #push!(code_parts, "$(ptr_name)= complex_to_float_zerocopy($ptr_name)")
                push!(code_parts, "idx = Vec{$(2*n_elems),Int64}($idx_tuple)")
                push!(code_parts, "v = vgather($(ptr_name), idx)")
                
            else
                # Multiple vgathers for large radix
                #push!(code_parts, "$(ptr_name)= complex_to_float_zerocopy($ptr_name)")
                
                for chunk_start in 1:complexes_per_vec:n_elems
                    chunk_end = min(chunk_start + complexes_per_vec - 1, n_elems)
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
    else # mode = :default
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

# Updated load_gen for SIMD vector loading
function load_gen_simd(x_vars; mode, T, ptr_name="px", SIMD_BITS=256)
    n = length(x_vars)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    n_floats = 2 * n
    
    # Extract indices from variable names
    indices = Int[]
    for var in x_vars
        m = match(r"(\d+)", var)
        push!(indices, parse(Int, m.captures[1]))
    end
    
    # Check if contiguous
    is_contiguous = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if is_contiguous && n <= complexes_per_vec
        # Contiguous load - single vload
        offset = 2 * (indices[1] - 1) * sizeof(T)
        return """
        # Load $n contiguous complex numbers as Vec{$(n_floats),$T}
        LANE = SIMD.VecRange{$n_floats}(0)
        v_all = $ptr_name[LANE + $offset]
        #v_all = vload(Vec{$(n_floats),$T}, $ptr_name, 1 + $offset)
        """
    elseif n <= complexes_per_vec
        # Non-contiguous - use vgather
        float_indices = Int[]
        for idx in indices
            push!(float_indices, 2*idx - 1)
            push!(float_indices, 2*idx)
        end
        idx_tuple = Tuple(float_indices)
        
        return """
        # Gather $n non-contiguous complex numbers
        idx = Vec($idx_tuple)
        v_all = vgather($ptr_name, idx)
        """
    else
        # Multiple vectors needed
        code_parts = String[]
        for chunk_start in 1:complexes_per_vec:n
            chunk_end = min(chunk_start + complexes_per_vec - 1, n)
            chunk_size = chunk_end - chunk_start + 1
            chunk_indices = indices[chunk_start:chunk_end]
            
            float_indices = Int[]
            for idx in chunk_indices
                push!(float_indices, 2*idx - 1)
                push!(float_indices, 2*idx)
            end
            
            chunk_id = (chunk_start - 1) ÷ complexes_per_vec + 1
            idx_tuple = Tuple(float_indices)
            
            push!(code_parts, "idx$chunk_id = Vec($idx_tuple)")
            push!(code_parts, "v$chunk_id = vgather($ptr_name, idx$chunk_id)")
        end
        
        return join(code_parts, "\n")
    end
end

# Updated store_gen for SIMD vector storing
function store_gen_simd(y_vars, src_vars; mode, T, ptr_name="py", SIMD_BITS=256)
    n = length(y_vars)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    n_floats = 2 * n
    
    # Extract indices from y variable names
    indices = Int[]
    for var in y_vars
        m = match(r"\[(\d+)\]", var)
        if m !== nothing
            push!(indices, parse(Int, m.captures[1]))
        end
    end
    
    # Check if contiguous
    is_contiguous = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if is_contiguous && n <= complexes_per_vec
        # Contiguous store - use vstore
        offset = 2 * (indices[1] - 1) * sizeof(T)
        
        # Handle special cases for src_vars
        if src_vars == ["+", "-"]
            return """
            # Store sum and diff
            vstore(v1 + v2, $ptr_name, $offset)
            vstore(v1 - v2, $ptr_name, $(offset + n_floats * sizeof(T)))
            """
        else
            # Build vector from source variables
            vals = String[]
            for src in src_vars
                if endswith(src, "_r") || endswith(src, "_i")
                    push!(vals, src)
                else
                    push!(vals, "$(src)_r", "$(src)_i")
                end
            end
            vals_tuple = "(" * join(vals, ",") * ")"
            
            return """
            # Store $n contiguous complex numbers
            v_out = Vec{$(n_floats),$T}($vals_tuple)
            vstore(v_out, $ptr_name, $offset)
            """
        end
    elseif n <= complexes_per_vec
        # Non-contiguous - use vscatter
        float_indices = Int[]
        for idx in indices
            push!(float_indices, 2*idx - 1)
            push!(float_indices, 2*idx)
        end
        
        vals = String[]
        for src in src_vars
            push!(vals, "$(src)_r", "$(src)_i")
        end
        
        idx_tuple = "(" * join(float_indices, ",") * ")"
        vals_tuple = "(" * join(vals, ",") * ")"
        
        return """
        # Scatter $n non-contiguous complex numbers
        idx = Vec{$(n_floats),Int64}($idx_tuple)
        v_out = Vec{$(n_floats),$T}($vals_tuple)
        vscatter(v_out, $ptr_name, idx)
        """
    else
        # Multiple vscatters for large radix
        code_parts = String[]
        for chunk_start in 1:complexes_per_vec:n
            chunk_end = min(chunk_start + complexes_per_vec - 1, n)
            chunk_size = chunk_end - chunk_start + 1
            chunk_indices = indices[chunk_start:chunk_end]
            chunk_src = src_vars[chunk_start:chunk_end]
            
            float_indices = Int[]
            for idx in chunk_indices
                push!(float_indices, 2*idx - 1)
                push!(float_indices, 2*idx)
            end
            
            vals = String[]
            for src in chunk_src
                push!(vals, "$(src)_r", "$(src)_i")
            end
            
            chunk_id = (chunk_start - 1) ÷ complexes_per_vec + 1
            idx_tuple = "(" * join(float_indices, ",") * ")"
            vals_tuple = "(" * join(vals, ",") * ")"
            
            push!(code_parts, "idx$chunk_id = Vec{$(2*chunk_size),Int64}($idx_tuple)")
            push!(code_parts, "v_out$chunk_id = Vec{$(2*chunk_size),$T}($vals_tuple)")
            push!(code_parts, "vscatter(v_out$chunk_id, $ptr_name, idx$chunk_id)")
        end
        
        return join(code_parts, "\n")
    end
    
end

# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int, suffixes::SuffixFlags, D::AbstractArray{String}, p::Int, op, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat
    global inc = inccounter()
    
    # ALL ME
    mode = :default
    
    has_y = has_flag(suffixes, Y)
    #has_vec = has_flag(suffixes, VEC)  # VEC flag indicates final stage

    input = op.eo ? "y" : "x"
    output = !has_y && op.eo ? "x" : "y"
    
    # Key parameters for Stockham algorithm
    radix = n
    stride = op.stride
    n_groups = op.n_groups
    input_spacing = SIZE ÷ radix  # Spacing between input elements in each butterfly
    
    # INPUT indexing: Always strided by input_spacing
    x = mode == :default ? ["$(input)$(p + 1 + (i-1)*input_spacing)" for i in 1:radix] : ["v_all$(p + 1 + (i-1)*input_spacing)" for i in 1:radix] 
    #TODO Convert "x" input values to "v$i" / "v_all" respectible vector capacities as constructed by load_gen_simd lamda
    
    # OUTPUT indexing: Depends on whether this is the final stage
    base = (p ÷ stride) * (stride * radix) + (p % stride)
    y = ["$output[$(base + 1 + i*stride)]" for i in 0:radix-1]
    
    d = D == String[] ? nothing : D
    
    px = mode == :vgather ? "p$(input) = reinterpret($T, $(input));" : "" 
    #py = unsafe_load_mode ? "$(output) = reinterpret($T, $(prev_output));" : ""
    py = ""

    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits

    # MOST IMPORTANT LINE!!!
    kernel_code = mode == :default ? recfft2(y, x, d, nothing, true, T, 1, mode, py) : recfft2_simd(y, x, d, nothing, true, T, 1, mode, py, complexes_per_vec) 
    kernel_code = "$px\n$kernel_code"
    
    #TODO Instead of returning Complex{$T} make all kernels, both scalar and vectorized return reinterpret(T, y) outputs. Let external kernels handle this
    
    if isempty(kernel_code)
        return quote end
    else
        try
            parsed_expr = Meta.parse("begin\n$kernel_code\nend")
            @show parsed_expr
            return parsed_expr
        catch e
            @warn "Failed to parse kernel code: $e"
            @warn "Kernel code was: $kernel_code"
            return quote
                copyto!(y, x) # Return wrong, yet functionl pass to next layer
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

# GROUP THEORY AUTOMORHISM FOR GF()
#=
function map_to_groups(numbers::AbstractArray{Int}, MODULO::Int)
  return ((numbers .- 1) .÷ MODULO) .+ 1
end
=#

#TODO SIMPIFY CISPI STRING AND NEW PARSER
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
    elseif w == "-1"
        return "-$(tmp)_r, -$(tmp)_i"
    elseif w == "-im"
        return "$(tmp)_i, -$(tmp)_r"
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

#TODO: Stick to one scalar sat expr vocabulary
function sat_expr(sign, x1, x2, w)
  is_t = startswith(x1, "t") || startswith(x2, "t")
  if w == "1"
      #return is_t ? 
          "$(x1)_r $sign $(x2)_r, $(x1)_i $sign $(x2)_i" #:
          #"$x1[2] $sign $x2[2], $x2[1] $sign $x1[1]"
  elseif w == "-1"
          "-($(x1)_r $sign $(x2)_r), -($(x1)_i $sign $(x2)_i)" #:
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

# Scalar version of the recursive radix generator for n = 2^q sizes
function recfft2(y, x, d, w, root, ::Type{T}, tmp_base=1, mode=:default, py="") where T <: AbstractFloat
  n = length(x)

  if n == 1
    ""
  elseif n == 2
    s = if !isnothing(d)
          if isnothing(w)
            if root
                load_real_imag_gen(x; mode=mode, T=T) * "\n" * 
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" * """
                $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(sat_expr("tmp0", "$(d[1])")))
                """
            end
          end
        else
          if root
            load_real_imag_gen(x; mode=mode, T=T) * "\n" * """
            $(y[1]), $(y[2]) = Complex{$T}($(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i), Complex{$T}($(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i)
            """ 
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
           s3p = "$py" * "\n" * "$(tmp_decls)" * "\n" *
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
          s3p = "$py" * "\n" * "$(y[1])" * foldl(*, vmap(i -> ",$(y[i])", 2:n2)) *
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
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i" * foldl(*, vmap(i -> ", $(y[i+n2])_r, $(y[i+n2])_i", 2:n2)) *
              " = " *
              "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", "$(w[n2+1])"))" *
              foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", "$(w[n2+i])"))", 2:n2)) * "\n"
    end
  end
  end
  s = n == 4 ? load_real_imag_gen(x; mode=mode, T=T) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
  return s
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

# Core SIMD saturated arithmetic for complex numbers stored as [r1,i1,r2,i2,...]
function sat_expr_simd_vec(sign::String, v1_name::String, v2_name::String, w::String, ::Type{T}, n_complex::Int) where T <: AbstractFloat
    """
    Perform complex butterfly with twiddle: (v1 ± v2) * w
    Vectors store complex as interleaved: [r1,i1,r2,i2,...]
    """
    
    if w == "1"
        # Simple add/subtract
        op = sign == "+" ? "+" : "-"
        return """
        Vec{$(2*n_complex),$T}($v1_name $op $v2_name)
        """
        
    elseif w == "-im"
        # -i * (v1 ± v2) = ±(i1 ∓ i2) ± i*(r1 ∓ r2)
        # Result: swap real/imag and negate appropriately
        op = sign == "+" ? "+" : "-"
        return """
        begin
            diff = $v1_name $op $v2_name
            # Swap real/imag: [r,i] -> [i,-r] for -im multiplication
            shufflevector(diff, Val($(join([2*i-1 for i in 1:n_complex], ",")))) * 
                Vec{$(2*n_complex),$T}($((join(["1" for _ in 1:n_complex], ",")))) -
            shufflevector(diff, Val($(join([2*i-2 for i in 1:n_complex], ",")))) * 
                Vec{$(2*n_complex),$T}($((join(["1" for _ in 1:n_complex], ","))))
        end
        """
        
    elseif w == "INV_SQRT2_Q4"
        # (v1 ± v2) * (1-i)/√2 = [(r±s+i±t)/√2, (i±t-r∓s)/√2]
        op = sign == "+" ? "+" : "-"
        return """
        begin
            diff = $v1_name $op $v2_name
            r_plus_i = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ",")))) + 
                       shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ","))))
            i_minus_r = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ",")))) - 
                        shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
            # Interleave back: [r1,i1,r2,i2,...]
            shufflevector(r_plus_i, i_minus_r, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ",")))) * 
                Vec{$(2*n_complex),$T}(($(join(["INV_SQRT2" for _ in 1:2*n_complex], ","))))
        end
        """
        
    elseif w == "-INV_SQRT2_Q1"
        # -(v1 ± v2) * (1+i)/√2 = [-(r±s-i∓t)/√2, -(r±s+i±t)/√2]
        op = sign == "+" ? "+" : "-"
        return """
        begin
            diff = $v1_name $op $v2_name
            i_minus_r = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ",")))) - 
                        shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
            neg_r_plus_i = -(shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ",")))) + 
                            shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ",")))))
            shufflevector(i_minus_r, neg_r_plus_i, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ",")))) * 
                Vec{$(2*n_complex),$T}(($(join(["INV_SQRT2" for _ in 1:2*n_complex], ","))))
        end
        """
        
    else
        # General CISPI twiddle factors
        parsed = parse_cispi(w)
        num, den, is_q1 = parsed.num, parsed.den, parsed.q1
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"
        
        # Build cosine and sine vectors (broadcast to all complex numbers)
        cos_vec = "Vec{$(2*n_complex),$T}(($(join(["$c" for _ in 1:2*n_complex], ","))))"
        sin_vec = "Vec{$(2*n_complex),$T}(($(join(["$s" for _ in 1:2*n_complex], ","))))"
        
        if startswith(w, "CISPI")
            # (a+bi)(c+di) = (ac-bd) + i(ad+bc)
            # Q1: cos+i*sin, Q4: cos-i*sin
            sign_s = is_q1 ? "+" : "-"
            op = sign == "+" ? "+" : "-"
            
            return """
            begin
                diff = $v1_name $op $v2_name
                # Extract real and imag parts
                real_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
                imag_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ","))))
                
                # Complex multiply: (r+ii) * (c±is)
                # Result real: r*c ∓ i*s
                # Result imag: r*s ± i*c
                res_real = muladd(real_parts, $cos_vec[0:$(n_complex-1)], 
                                  $(is_q1 ? "-" : "+")imag_parts * $sin_vec[0:$(n_complex-1)])
                res_imag = muladd(real_parts, $sin_vec[0:$(n_complex-1)], 
                                  $(is_q1 ? "+" : "-")imag_parts * $cos_vec[0:$(n_complex-1)])
                
                # Interleave back
                shufflevector(res_real, res_imag, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ","))))
            end
            """
            
        elseif startswith(w, "-im*CISPI")
            # -i*(cos±i*sin) = ±sin - i*cos
            op = sign == "+" ? "+" : "-"
            sign_real = is_q1 ? "+" : "-"
            
            return """
            begin
                diff = $v1_name $op $v2_name
                real_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
                imag_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ","))))
                
                # -i * (cos±i*sin) = ±sin - i*cos
                res_real = muladd($(sign_real == "+" ? "" : "-")real_parts, $sin_vec[0:$(n_complex-1)], 
                                  imag_parts * $cos_vec[0:$(n_complex-1)])
                res_imag = muladd(-real_parts, $cos_vec[0:$(n_complex-1)], 
                                  $(sign_real == "+" ? "" : "-")imag_parts * $sin_vec[0:$(n_complex-1)])
                
                shufflevector(res_real, res_imag, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ","))))
            end
            """
            
        elseif startswith(w, "-CISPI")
            # -(cos±i*sin)
            op = sign == "+" ? "+" : "-"
            sign_s = is_q1 ? "-" : "+"
            
            return """
            begin
                diff = $v1_name $op $v2_name
                real_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
                imag_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ","))))
                
                res_real = muladd(-real_parts, $cos_vec[0:$(n_complex-1)], 
                                  $(sign_s)imag_parts * $sin_vec[0:$(n_complex-1)])
                res_imag = muladd(-real_parts, $sin_vec[0:$(n_complex-1)], 
                                  $(is_q1 ? "-" : "+")imag_parts * $cos_vec[0:$(n_complex-1)])
                
                shufflevector(res_real, res_imag, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ","))))
            end
            """
            
        elseif startswith(w, "im*CISPI")
            # i*(cos±i*sin) = ∓sin + i*cos
            op = sign == "+" ? "+" : "-"
            sign_real = is_q1 ? "-" : "+"
            
            return """
            begin
                diff = $v1_name $op $v2_name
                real_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==0], ","))))
                imag_parts = shufflevector(diff, Val($(join([i for i in 0:2*n_complex-1 if i%2==1], ","))))
                
                res_real = muladd($(sign_real)real_parts, $sin_vec[0:$(n_complex-1)], 
                                  -imag_parts * $cos_vec[0:$(n_complex-1)])
                res_imag = muladd(real_parts, $cos_vec[0:$(n_complex-1)], 
                                  $(sign_real)imag_parts * $sin_vec[0:$(n_complex-1)])
                
                shufflevector(res_real, res_imag, Val($(join(vcat([[2*i,2*i+n_complex] for i in 0:n_complex-1]...), ","))))
            end
            """
        end
    end
end

# Scalar version 1: Apply twiddle to single temp variable (tmp already computed)
function sat_expr_simd_scalar(tmp::String, w::String, ::Type{T}) where T <: AbstractFloat
    if w == "1"
        return "$(tmp)_r, $(tmp)_i"
    elseif w == "-im"
        return "$(tmp)_i, -$(tmp)_r"
    elseif w == "INV_SQRT2_Q4"
        return "INV_SQRT2*($(tmp)_r + $(tmp)_i), INV_SQRT2*($(tmp)_i - $(tmp)_r)"
    elseif w == "-INV_SQRT2_Q1"
        return "INV_SQRT2*($(tmp)_i - $(tmp)_r), -INV_SQRT2*($(tmp)_r + $(tmp)_i)"
    else
        parsed = parse_cispi(w)
        c = "COSPI_$(parsed.num)_$(parsed.den)"
        s = "SINPI_$(parsed.num)_$(parsed.den)"
        
        if startswith(w, "CISPI")
            if parsed.q1
                return "muladd($c, $(tmp)_r, -$s * $(tmp)_i), muladd($s, $(tmp)_r, $c * $(tmp)_i)"
            else
                return "muladd($c, $(tmp)_r, $s * $(tmp)_i), muladd(-$s, $(tmp)_r, $c * $(tmp)_i)"
            end
        elseif startswith(w, "-im*CISPI")
            if parsed.q1
                return "muladd($s, $(tmp)_r, $c * $(tmp)_i), muladd(-$c, $(tmp)_r, $s * $(tmp)_i)"
            else
                return "muladd(-$s, $(tmp)_r, $c * $(tmp)_i), muladd(-$c, $(tmp)_r, -$s * $(tmp)_i)"
            end
        elseif startswith(w, "-CISPI")
            if parsed.q1
                return "muladd(-$c, $(tmp)_r, $s * $(tmp)_i), muladd(-$s, $(tmp)_r, -$c * $(tmp)_i)"
            else
                return "muladd(-$c, $(tmp)_r, -$s * $(tmp)_i), muladd($s, $(tmp)_r, -$c * $(tmp)_i)"
            end
        elseif startswith(w, "im*CISPI")
            if parsed.q1
                return "muladd(-$s, $(tmp)_r, -$c * $(tmp)_i), muladd($c, $(tmp)_r, -$s * $(tmp)_i)"
            else
                return "muladd($s, $(tmp)_r, -$c * $(tmp)_i), muladd($c, $(tmp)_r, $s * $(tmp)_i)"
            end
        end
    end
end

# Scalar version 2: Butterfly operation with twiddle: (x1 ± x2) * w
function sat_expr_simd_scalar(sign::String, x1::String, x2::String, w::String, ::Type{T}) where T <: AbstractFloat
    if w == "1"
        return "$(x1)_r $sign $(x2)_r, $(x1)_i $sign $(x2)_i"
        
    elseif w == "-im"
        # -i*(x1 ± x2) = ±(i1 ∓ i2) ± i*(r1 ∓ r2)
        return "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r"
        
    elseif w == "INV_SQRT2_Q4"
        # (x1 ± x2) * (1-i)/√2
        return "INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i)), " *
               "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r))"
               
    elseif w == "-INV_SQRT2_Q1"
        # -(x1 ± x2) * (1+i)/√2
        return "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r)), " *
               "-INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i))"
               
    else
        parsed = parse_cispi(w)
        c = "COSPI_$(parsed.num)_$(parsed.den)"
        s = "SINPI_$(parsed.num)_$(parsed.den)"
        
        if startswith(w, "CISPI")
            if parsed.q1
                # Q1: cos + i*sin
                return "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))"
            else
                # Q4: cos - i*sin
                return "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))"
            end
            
        elseif startswith(w, "-im*CISPI")
            if parsed.q1
                # -i*(cos + i*sin) = sin - i*cos
                return "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))"
            else
                # -i*(cos - i*sin) = -sin - i*cos
                return "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))"
            end
            
        elseif startswith(w, "-CISPI")
            if parsed.q1
                # -cos - i*sin
                return "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))"
            else
                # -cos + i*sin
                return "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))"
            end
            
        elseif startswith(w, "im*CISPI")
            if parsed.q1
                # i*(cos + i*sin) = -sin + i*cos
                return "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))"
            else
                # i*(cos - i*sin) = sin + i*cos
                return "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
                       "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))"
            end
        end
    end
end

# Complete SIMD FFT kernel generator
"""
recfft2_simd(y, x, d, w, root, ::Type{T}; tmp_base=1, mode=:default, py="", complexes_per_vec=2)

A SIMD-aware code-generator replacement for your scalar recfft2 codegen.
- Produces vector temporaries tvec1, tvec2, ... each of type Vec{2*complexes_per_vec, T}
  (2 floats per complex: real, imag interleaved).
- `complexes_per_vec` = number of complex elements per SIMD register (e.g. 2 for AVX2 + Complex{Float32}).
- Returns source string for generated kernel (same style as your scalar generator).
"""

# VERY DIFFICULT

#TODO To do a horizontal unrolling of all possible saturated scalar 't' vals of recfft2 to a single ymm/zmm vectorized operation,
# the solution can't be concluded at the terminal n = 2 level. Optimization can fix even at n = 32 of AVX512 Real Float16 zmm registers... (16 * 32 = 512)
# Do a horizontal expr saturator FUNCTOR anamorphism on top of symbolic SIMD width ymms produced at each height... pipe? (|)
# This way we produce cross-platform cross-type cross-size saturated vectorized ops...
function recfft2_simd(y, x, d, w, root, ::Type{T}, tmp_base=1, mode=:vgather, py="", complexes_per_vec=2) where T <: AbstractFloat
    n = length(x)
    
    @show y, x, d, w, root
    
    if n == 1
        return ""
        
    elseif n == 2
        # Base case: 2-point butterfly
        if !isnothing(d)
            if isnothing(w) && root
                # Root FFT2 with D matrix - use scalar operations for now
                return """
                x1_r, x1_i = real($(x[1])), imag($(x[1]))
                x2_r, x2_i = real($(x[2])), imag($(x[2]))
                sum_r, sum_i = x1_r + x2_r, x1_i + x2_i
                diff_r, diff_i = x1_r - x2_r, x1_i - x2_i
                $(y[1]) = Complex{$T}(sum_r, sum_i)
                #$(y[2]) = Complex{$T}($(sat_expr_simd_scalar("diff", d[1], T)))
                """
            end
        else
            if root
                return """
                x1_r, x1_i = real($(x[1])), imag($(x[1]))
                x2_r, x2_i = real($(x[2])), imag($(x[2]))
                $(y[1]) = Complex{$T}(x1_r + x2_r, x1_i + x2_i)
                #$(y[2]) = Complex{$T}(x1_r - x2_r, x1_i - x2_i)
                """
            else
                # Non-root 2-point
                if isnothing(w)
                    @show y
                    @show y[1]
                    return """
                    $(y[1])_r, $(y[1])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i
                    $(y[1])_r, $(y[1])_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
                    """
                else
                    # With twiddle factors
                    if w[1] == "1"
                        #twiddle1 = "$(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i"
                    else
                        #twiddle1 = sat_expr_simd_scalar("+", "$(x[1])", "$(x[2])", w[1], T)
                    end
                    #twiddle2 = sat_expr_simd_scalar("-", "$(x[1])", "$(x[2])", w[2], T)
                    lhs = sat_expr_simd_vec("+", "$(x[1])", "$(x[2])", "$(w[1])", T, n)

                    return  """
                    $(y[1]) = $lhs
                    """
                end
            end
        end
        
    else
        # Recursive decomposition
        #t = ["t$i" for i in tmp_base:tmp_base + n - 1]

        # compute how many vector temporaries (Vec slots) are needed per half
        cpv = complexes_per_vec
        n2 = n ÷ 2
        vecs_per_half = ceil(Integer, n2 / cpv)   # how many Vec slots per half
        
        # total vector temporaries (first half + second half)
        total_tvecs = 2 * vecs_per_half
        
        # create t as vector temporaries: total = total_tvecs, starting at tmp_base
        t = ["t$(i)" for i in tmp_base:(tmp_base + total_tvecs - 1)]
        @show t
        new_tmp_base = tmp_base + total_tvecs
        @show new_tmp_base
        
        # If you want the chunk ranges for other uses, build them like this:
        chunks = [ ((k-1)*cpv + 1) : min(k*cpv, n2) for k in 1:vecs_per_half ]
        @show chunks
        
        # Pass the first vecs_per_half vector temporaries to the 'even' recursion
        s1 = recfft2_simd(t[1:vecs_per_half], x[1:2:n], nothing, nothing, false, T, new_tmp_base, mode, py, complexes_per_vec)
        @show s1
        
        # Pass the next vecs_per_half vector temporaries to the 'odd' recursion
        s2 = recfft2_simd(t[vecs_per_half+1:2*vecs_per_half], x[2:2:n], nothing,
                  get_twiddle_expression(collect(0:n2-1), n; T=T, accuracy=nothing),
                  false, T, new_tmp_base, mode, py, complexes_per_vec)
        @show s2
        
        # Combine results with butterfly operations
        # TODO Place symbolic saturator for wider perms here???
        s3p, s3m = generate_butterfly_combination(y, t, d, w, n, n2, root, T, py)
        @show s3p, s3m

        # Add load operations at the base recursion level
        load_code = n == 4 ? load_gen_simd(x; mode=mode, T=T, ptr_name="px", SIMD_BITS=SIMD_BITS) * "\n" : ""
        
        return load_code * s1 * s2 * s3p * s3m
    end
end

# Helper to generate butterfly combination code
function generate_butterfly_combination(y, t, d, w, n, n2, root, ::Type{T}, py) where T
    # Generate temporary variables for sum/diff
    @show y, t, d, w, n, n2
    tmp_decls = if n > 2
        parts = String[]
        for i in 2:n2
            push!(parts, "tmp$(i-2)_r, tmp$(i-2)_i = $(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i")
            push!(parts, "tmp$(i-2+n2)_r, tmp$(i-2+n2)_i = $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i")
        end
        join(parts, "\n") * "\n"
    else
        ""
    end
    
    # Generate output assignments
    if !isnothing(d)
        # With D matrix
        s3p = root ? "$py\n$tmp_decls$(y[1]) = Complex{$T}($(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i)\n" : ""
        for i in 2:n2
            s3p *= "$(y[i]) = Complex{$T}($(sat_expr_simd_scalar("tmp$(i-2)", d[i-1], T)))\n"
        end
        
        s3m = "$(y[n2+1]) = Complex{$T}($(sat_expr_simd_scalar("-", "$(t[1])", "$(t[1+n2])", d[n2], T)))\n"
        for i in 2:n2
            s3m *= "$(y[i+n2]) = Complex{$T}($(sat_expr_simd_scalar("tmp$(i-3+n2)", d[i+n2-1], T)))\n"
        end
        
    elseif !isnothing(w)
        # With twiddle factors
        s3p = "$tmp_decls"
        
        # For first element: directly compute (t[1] + t[1+n2]) * w[1]
        if w[1] == "1"
            s3p *= "$(y[1])_r, $(y[1])_i = $(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i\n"
        else
            s3p *= "$(y[1])_r, $(y[1])_i = $(sat_expr_simd_scalar("+", "$(t[1])", "$(t[1+n2])", w[1], T))\n"
        end
        
        # For remaining elements: use precomputed tmp values
        for i in 2:n2
            s3p *= "$(y[i])_r, $(y[i])_i = $(sat_expr_simd_scalar("tmp$(i-2)", w[i], T))\n"
        end
        
        # Minus part: (t[1] - t[1+n2]) * w[n2+1]
        s3m = "$(y[n2+1])_r, $(y[n2+1])_i = $(sat_expr_simd_scalar("-", "$(t[1])", "$(t[1+n2])", w[n2+1], T))\n"
        for i in 2:n2
            s3m *= "$(y[i+n2])_r, $(y[i+n2])_i = $(sat_expr_simd_scalar("tmp$(i-3+n2)", w[n2+i], T))\n"
        end
        
    else
        # Simple butterfly
        if root
            s3p = "$py\n"
            for i in 1:n2
                s3p *= "$(y[i]) = Complex{$T}($(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i)\n"
            end
            s3m = ""
            for i in 1:n2
                s3m *= "$(y[n2+i]) = Complex{$T}($(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i)\n"
            end
        else
            s3p = ""
            for i in 1:n2
                s3p *= "$(y[i])_r, $(y[i])_i = $(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i\n"
            end
            s3m = ""
            for i in 1:n2
                s3m *= "$(y[n2+i])_r, $(y[n2+i])_i = $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i\n"
            end
        end
    end
    
    return s3p, s3m
end

### MODEL RADIX. VERY SMART


# Vectorized XOR operation to flip sign at chosen vector elements
@inline function signflip(v::Vec{N,T}, mask::Vec{N,UInt32}) where {N, T<:AbstractFloat}
    # Reinterpret float as uint, XOR with sign bit, reinterpret back
    v_uint = reinterpret(Vec{N,UInt32}, v)
    v_flipped = v_uint ⊻ mask
    return reinterpret(Vec{N,T}, v_flipped)
end

@inline function vfft4_xor(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = x[LANE + 1]
        
        lo = shufflevector(v, Val((0, 1, 2, 3)))
        hi = shufflevector(v, Val((4, 5, 6, 7)))
        
        add_vec = lo + hi
        sub_vec = lo - hi
        
        # Apply twiddle with XOR sign flip (flip 4th element only)
        sign_mask = Vec{4,UInt32}((0x00000000, 0x00000000, 0x00000000, 0x80000000))
        sub_vec_shuffled = shufflevector(sub_vec, Val((0, 1, 3, 2)))
        sub_vec = signflip(sub_vec_shuffled, sign_mask)
        
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))
        
        y12 = t12 + t34
        y34 = t12 - t34
        
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        y[LANE + 1] = OUT
    end
end