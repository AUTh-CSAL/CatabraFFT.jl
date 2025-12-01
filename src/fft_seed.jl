include("suffix.jl")
include("radix_plan.jl")
using SIMD

const SIMD_BITS = 256


#TODO 1. Fix SIMD y->y being hard-coded as x->y 2. y[1] -> y1 for y->y layers!!! ZZZZZ

# Twiddle factor quadrant enum for efficient representation
@enum TwiddleQuadrant::Int8 begin
    Q1 = 1      # cispi(num/den): cos + i*sin
    Q4 = 4      # cispi(-num/den): cos - i*sin
    NegQ1 = -1  # -cispi(num/den): -cos - i*sin
    NegQ4 = -4  # -cispi(-num/den): -cos + i*sin
    ImQ1 = 11   # im*cispi(num/den): -sin + i*cos
    ImQ4 = 14   # im*cispi(-num/den): sin + i*cos
    NegImQ1 = -11  # -im*cispi(num/den): sin - i*cos
    NegImQ4 = -14  # -im*cispi(-num/den): -sin - i*cos
end

# Twiddle factor as (numerator, denominator, quadrant)
const Twiddle = Tuple{Int, Int, TwiddleQuadrant}

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
load_real_imag_gen = (t; mode, T, input="x") -> begin
    vec_width = 2sizeof(T)  
    
    join([
        let
            m = match(r"(\d+)\D*$", s)
            num = parse(Int, m.captures[1])
            var = startswith(s, "x") ? "x" :
                  startswith(s, "y") ? "y" :
                  startswith(s, "D") ? "d" : error("Unknown input: $s")
            prefix = i == 1 ? "" : " "
            "$(prefix)$(var)$(num)_r , $(prefix)$(var)$(num)_i = $(input)[$(2*num-1)], $(input)[$(2*num)]"
        end
        for (i, s) in enumerate(t)
    ], "; ")
end

# Updated load_gen for SIMD vector loading
function load_gen_simd(x_vars; mode, T, ptr_name="x", SIMD_BITS=256)
    n = length(x_vars)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    n_floats = 2 * n
    
    # Extract indices from variable names (x1 -> 1, x5 -> 5, etc.)
    indices = [parse(Int, match(r"(\d+)", var).captures[1]) for var in x_vars]
    
    # Check if contiguous
    is_contiguous = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if is_contiguous && n <= complexes_per_vec
        # Case 1: Contiguous access - single vload
        start_idx = 2 * (indices[1] - 1) + 1  # Convert to 1-based float index
        return "l_all = vload(Vec{$(n_floats),$T}, $ptr_name, $start_idx)"
        
    elseif n <= complexes_per_vec
        # Case 2: Non-contiguous, fits in one register - single vgather
        float_indices = Int[]
        for idx in indices
            push!(float_indices, 2 * (idx - 1) + 1)  # Real part (1-based)
            push!(float_indices, 2 * (idx - 1) + 2)  # Imag part (1-based)
        end
        idx_tuple = tuple(float_indices...)
        
        return "l_all = vgather($ptr_name, Vec($idx_tuple))"
        
    else
        # Case 3: Large discrete access - multiple vgathers or vloads
        code_lines = String[]
        n_chunks = cld(n, complexes_per_vec)  # Ceiling division

        for chunk_id in 1:n_chunks
            chunk_start = (chunk_id - 1) * complexes_per_vec + 1
            chunk_end = min(chunk_id * complexes_per_vec, n)
            chunk_indices = indices[chunk_start:chunk_end]
            chunk_n_floats = 2 * length(chunk_indices)

            # Check if this chunk is contiguous
            chunk_is_contiguous = all(i -> chunk_indices[i] == chunk_indices[1] + i - 1, 2:length(chunk_indices))

            if chunk_is_contiguous
                # Use vload for contiguous chunk
                start_idx = 2 * (chunk_indices[1] - 1) + 1
                push!(code_lines, "l$chunk_id = vload(Vec{$chunk_n_floats,$T}, $ptr_name, $start_idx)")
            else
                # Use vgather for non-contiguous chunk
                float_indices = Int[]
                for idx in chunk_indices
                    push!(float_indices, 2 * (idx - 1) + 1)
                    push!(float_indices, 2 * (idx - 1) + 2)
                end
                idx_tuple = tuple(float_indices...)
                push!(code_lines, "l$chunk_id = vgather($ptr_name, Vec($idx_tuple))")
            end
        end

        return join(code_lines, "\n")
    end
end

# Updated store_gen for SIMD vector storing
function store_gen_simd(y_vars, src_vars; mode, T, ptr_name="y", SIMD_BITS=256)
    n = length(y_vars)
    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits
    n_floats = length(src_vars)  # Use actual number of floats passed
    
    # Extract indices from y variable names (y[1] -> 1, y[5] -> 5, etc.)
    indices = [parse(Int, match(r"\[(\d+)\]", var).captures[1]) for var in y_vars]
    
    # Check if contiguous
    is_contiguous = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
    
    if is_contiguous && n <= complexes_per_vec
        # Case 1: Contiguous store - single vstore
        start_idx = 2 * (indices[1] - 1) + 1  # 1-based float index
        vals_str = join(src_vars, ", ")

        # Use intermediate variable for clarity - wrap in begin/end
        return "begin\nv_out = Vec{$(n_floats),$T}($vals_str)\nvstore(v_out, $ptr_name, $start_idx)\nend"

    elseif n <= complexes_per_vec
        # Case 2: Non-contiguous, fits in one register - single vscatter
        float_indices = Int[]
        for idx in indices
            push!(float_indices, 2 * (idx - 1) + 1)
            push!(float_indices, 2 * (idx - 1) + 2)
        end

        idx_str = join(float_indices, ", ")
        vals_str = join(src_vars, ", ")

        # Use intermediate variable for clarity - wrap in begin/end
        return "begin\nv_out = Vec{$(n_floats),$T}($vals_str)\nvscatter(v_out, $ptr_name, Vec($idx_str))\nend"

    else
        # Case 3: Large discrete store - multiple vscatters or vstores
        code_lines = String[]
        n_chunks = cld(n, complexes_per_vec)
        floats_per_chunk = 2 * complexes_per_vec

        for chunk_id in 1:n_chunks
            chunk_start = (chunk_id - 1) * complexes_per_vec + 1
            chunk_end = min(chunk_id * complexes_per_vec, n)
            chunk_indices = indices[chunk_start:chunk_end]
            chunk_n_floats = 2 * length(chunk_indices)

            # Get corresponding source floats for this chunk
            src_start = (chunk_id - 1) * floats_per_chunk + 1
            src_end = min(chunk_id * floats_per_chunk, length(src_vars))
            chunk_src = src_vars[src_start:src_end]

            # Check if this chunk is contiguous
            chunk_is_contiguous = all(i -> chunk_indices[i] == chunk_indices[1] + i - 1, 2:length(chunk_indices))

            vals_str = join(chunk_src, ", ")

            if chunk_is_contiguous
                # Use vstore for contiguous chunk
                start_idx = 2 * (chunk_indices[1] - 1) + 1
                push!(code_lines, "v_out$chunk_id = Vec{$chunk_n_floats,$T}($vals_str)")
                push!(code_lines, "vstore(v_out$chunk_id, $ptr_name, $start_idx)")
            else
                # Use vscatter for non-contiguous chunk
                float_indices = Int[]
                for idx in chunk_indices
                    push!(float_indices, 2 * (idx - 1) + 1)
                    push!(float_indices, 2 * (idx - 1) + 2)
                end

                idx_str = join(float_indices, ", ")
                push!(code_lines, "v_out$chunk_id = Vec{$chunk_n_floats,$T}($vals_str)")
                push!(code_lines, "vscatter(v_out$chunk_id, $ptr_name, Vec($idx_str))")
            end
        end

        # Wrap multiple statements in begin/end block
        return "begin\n$(join(code_lines, "\n"))\nend"
    end
end

# Overload for when D is a type (no actual D matrix)
function makefftradix(n::Int, suffixes::SuffixFlags, ::Type{Vector{T}}, p::Int, op, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat
    # Call with empty vector when no D matrix is needed
    return makefftradix(n, suffixes, Union{String, Twiddle}[], p, op, SIZE, T, SIMD_BITS)
end

# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int, suffixes::SuffixFlags, D::Union{Vector{Union{String, Twiddle}}, Vector{T}}, p::Int, op, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat
    global inc = inccounter()
    
    mode = :vgather # ALL ME

    input, output = String(op.input_buffer), String(op.output_buffer)
    
    # Key parameters for Stockham algorithm
    radix = n 
    stride = op.stride
    input_spacing = SIZE ÷ radix  # Spacing between input elements in each butterfly
    
    # TODO FOR SOME n = s * n_g decompositions, input has wrong indexing  MUST DO!!!
    x = mode == :default ? ["$(input)$(p + 1 + (i-1)*input_spacing)" for i in 1:radix] : ["v$(p + 1 + (i-1)*input_spacing)" for i in 1:radix]

    is_terminal = has_flag(suffixes, VEC)

    if is_terminal
        # Terminal stage: use consecutive template indices (will be transformed later)
        base = 2 * ((p ÷ stride) * (stride * radix) + (p % stride))
        y = ["$output[$(base + 1 + i)]" for i in 0:2*radix-1]
    else
        # Non-terminal stage: DIF Stockham stores to consecutive blocks
        # Each sub-kernel p writes radix consecutive complexes starting at p*radix
        base_complex = p * radix
        y_indices = Int[]
        for k in 0:radix-1
            complex_idx = base_complex + k
            float_idx = 2 * complex_idx
            push!(y_indices, float_idx + 1)  
            push!(y_indices, float_idx + 2)  
        end
        y = ["$output[$(idx)]" for idx in y_indices]
    end
    
    d = isempty(D) ? nothing : D
    
    #px = mode == :vgather ? "p$(input) = reinterpret($T, $(input));" : "" 
    px = ""
    py = ""

    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits

    # Calculate vector width from complexes_per_vec
    # Each sub-kernel processes exactly 'radix' complex numbers
    floats_per_vec_full = 2 * complexes_per_vec
    floats_per_subkernel = 2 * n  # n is the radix
    floats_per_vec = min(floats_per_vec_full, floats_per_subkernel)

    # MOST IMPORTANT LINE!!!
    kernel_code = mode == :default ? recfft2(y, x, d, nothing, true, T, 1, mode, py, input) : recfft2_simd(y, x, d, nothing, true, T, floats_per_vec, 1, mode, py, complexes_per_vec, input, output)
    kernel_code = "$px\n$kernel_code"
    
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

# GROUP THEORY AUTOMORHISM FOR GF()
#=
function map_to_groups(numbers::AbstractArray{Int}, MODULO::Int)
  return ((numbers .- 1) .÷ MODULO) .+ 1
end
=#

function add_more_tmp_vars(x1, x2, wn, n)
    tmp_vars = String[]
    assignments = String[]
    idx = 0

    for i in 1:n
        real_plus = x1[2*i - 1]
        imag_plus = x1[2*i]
        push!(tmp_vars, "tmp$(idx)_r", "tmp$(idx)_i")
        push!(assignments, real_plus, imag_plus)
        idx += 1
    end

    for i in 1:n
        real_minus = x2[2*i - 1]
        imag_minus = x2[2*i]
        push!(tmp_vars, "tmp$(idx)_r", "tmp$(idx)_i")
        push!(assignments, real_minus, imag_minus)
        idx += 1
    end

    if !isempty(tmp_vars)
      return "$(join(tmp_vars, ", ")) = $(join(assignments, ", "))\n"
    end

    return ""
end

# Tuple-based sat_expr for (num, den, quadrant) format
function sat_expr(tmp, w)
    if w isa String
        if w == "1"
            return "$(tmp)_r, $(tmp)_i"
        elseif w == "-1"
            return "-$(tmp)_r, -$(tmp)_i"
        elseif w == "-im"
            return "$(tmp)_i, -$(tmp)_r"
        elseif w == "im"
            return "-$(tmp)_i, $(tmp)_r"
        elseif w == "INV_SQRT2_Q4"
            # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
            return "INV_SQRT2*($(tmp)_r + $(tmp)_i), " *
                   "INV_SQRT2*($(tmp)_i - $(tmp)_r)"
        elseif w == "-INV_SQRT2_Q1"
            # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ± b_i)/√2 ]
            return "INV_SQRT2*($(tmp)_i - $(tmp)_r), " *
                   "-INV_SQRT2*($(tmp)_r + $(tmp)_i)"
        end
            try
              # Meta.parse returns an expression, which needs to be evaluated/used
              # or passed to the Tuple logic. We'll evaluate it to get the value.
              w_parsed = Core.eval(Main, Meta.parse(w))
                  
              # If successfully parsed into a Tuple, jump to the Tuple logic below
              if w_parsed isa Tuple
                  return sat_expr(tmp, w_parsed)
              end
            catch
              # If parsing failed, it's a genuinely unknown string twiddle factor.
                error("Unknown string twiddle factor: $w")
            end

            error("Unknown string twiddle factor: $w")

    elseif w isa Tuple
        num, den, quadrant = w
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"

        if quadrant == Q1
            # cispi(num/den): cos + i*sin
            return "muladd($c, $(tmp)_r, -$s * $(tmp)_i), " *
                   "muladd($s, $(tmp)_r, $c * $(tmp)_i)"
        elseif quadrant == Q4
            # cispi(-num/den): cos - i*sin
            return "muladd($c, $(tmp)_r, $s * $(tmp)_i), " *
                   "muladd(-$s, $(tmp)_r, $c * $(tmp)_i)"
        elseif quadrant == NegQ1
            # -cispi(num/den): -cos - i*sin
            return "muladd(-$c, $(tmp)_r, $s * $(tmp)_i), " *
                   "muladd(-$s, $(tmp)_r, -$c * $(tmp)_i)"
        elseif quadrant == NegQ4
            # -cispi(-num/den): -cos + i*sin
            return "muladd(-$c, $(tmp)_r, -$s * $(tmp)_i), " *
                   "muladd($s, $(tmp)_r, -$c * $(tmp)_i)"
        elseif quadrant == ImQ1
            # im*cispi(num/den): -sin + i*cos
            return "muladd(-$s, $(tmp)_r, -$c * $(tmp)_i), " *
                   "muladd($c, $(tmp)_r, -$s * $(tmp)_i)"
        elseif quadrant == ImQ4
            # im*cispi(-num/den): sin + i*cos
            return "muladd($s, $(tmp)_r, -$c * $(tmp)_i), " *
                   "muladd($c, $(tmp)_r, $s * $(tmp)_i)"
        elseif quadrant == NegImQ1
            # -im*cispi(num/den): sin - i*cos
            return "muladd($s, $(tmp)_r, $c * $(tmp)_i), " *
                   "muladd(-$c, $(tmp)_r, $s * $(tmp)_i)"
        elseif quadrant == NegImQ4
            # -im*cispi(-num/den): -sin - i*cos
            return "muladd(-$s, $(tmp)_r, $c * $(tmp)_i), " *
                   "muladd(-$c, $(tmp)_r, -$s * $(tmp)_i)"
        else
        error("Unknown quadrant: $quadrant")
        end
    else
        error("Unknown twiddle type: $w")
    end
end

function sat_expr(sign, x1, x2, w)
    if w isa String 
        if w == "1"
            "$(x1)_r $sign $(x2)_r, $(x1)_i $sign $(x2)_i" 
        elseif w == "-1"
            "-($(x1)_r $sign $(x2)_r), -($(x1)_i $sign $(x2)_i)" 
        elseif w == "-im"
            # -i*(a ± b) = ±(b_i ∓ a_i) ± i*(b_r ∓ a_r)
            "$(x1)_i $sign $(x2)_i, $(x2)_r $sign $(x1)_r" 
        elseif w == "INV_SQRT2_Q4"
            # (a ± b) * (1-i)/√2 = [ (a_r ± b_r + a_i ± b_i)/√2 , (a_i ± b_i - a_r ∓ b_r)/√2 ]
            "INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i)), " *
            "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r))" 
        elseif w == "-INV_SQRT2_Q1"
            # -(a ± b) * (1+i)/√2 = [ -(a_r ± b_r - a_i ∓ b_i)/√2 , -(a_r ± b_r + a_i ± b_i)/√2 ]
            "INV_SQRT2*(($(x1)_i $sign $(x2)_i) - ($(x1)_r $sign $(x2)_r)), " *
            "-INV_SQRT2*(($(x1)_r $sign $(x2)_r) + ($(x1)_i $sign $(x2)_i))"
        end
    elseif w isa Tuple
        num, den, quadrant = w
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"
    
        if quadrant == Q1
            # Q1: cosθ + i sinθ
            "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
            "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))"
        elseif quadrant == Q4
            # Q4: cosθ - i sinθ
            "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
            "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i))"
        elseif quadrant == NegQ1
            # -cosθ - i sinθ
            "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i)), " *
            "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))" 
        elseif quadrant == NegQ4
            # -cosθ + i sinθ
            "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i)), " *
            "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i))" 
        elseif quadrant == ImQ1
            # i*(cosθ + i sinθ) = -sinθ + i cosθ
            "muladd(-$s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
            "muladd($c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))"
        elseif quadrant == ImQ4
            # i*(cosθ - i sinθ) = sinθ + i cosθ
            "muladd($s, $(x1)_r $sign $(x2)_r, -$c * ($(x1)_i $sign $(x2)_i)), " *
            "muladd($c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))" 
        elseif quadrant == NegImQ1
            # -i*(cosθ + i sinθ) = sinθ - i cosθ
            "muladd($s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
            "muladd(-$c, $(x1)_r $sign $(x2)_r, $s * ($(x1)_i $sign $(x2)_i))" 
        elseif quadrant == NegImQ4
            # -i*(cosθ - i sinθ) = -sinθ - i cosθ
            "muladd(-$s, $(x1)_r $sign $(x2)_r, $c * ($(x1)_i $sign $(x2)_i)), " *
            "muladd(-$c, $(x1)_r $sign $(x2)_r, -$s * ($(x1)_i $sign $(x2)_i))" 
        else
            error("Unkown quadrant: $quadrant")
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
function recfft2(y, x, d, w, root, ::Type{T}, tmp_base=1, mode=:default, py="", input_buffer="x") where T <: AbstractFloat
  n = length(x)

  if n == 1
    ""
  elseif n == 2
    s = if !isnothing(d)
          if isnothing(w)
            if root
                load_real_imag_gen(x; mode=mode, T=T, input=input_buffer) * "\n" *
                "tmp0_r, tmp0_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" * "\n" * "$py" * "\n" *
                "$(y[1]), $(y[2]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i\n" *
                "$(y[3]), $(y[4]) = $(sat_expr("tmp0", d[1]))"
            end
          end
        else
          if root
            load_real_imag_gen(x; mode=mode, T=T, input=input_buffer) * "\n" *
            "$(y[1]), $(y[2]), $(y[3]), $(y[4]) = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i, $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i" 
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
                $(y[1])_r, $(y[1])_i, $(y[2])_r, $(y[2])_i = $(sat_expr("+", "$(x[1])", "$(x[2])", "$(w[1])")), $(sat_expr("-", "$(x[1])", "$(x[2])", "$(w[2])"))
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
           lhs_p = "$(y[1]), $(y[2])" * foldl(*, vmap(i -> ", $(y[(2*i-1)]), $(y[(2*i)])", 2:n2))
           rhs_p = "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-2)", d[i-1]))", 2:n2))
           s3p = "$py" * "\n" * "$(tmp_decls)" * "\n" * lhs_p * " = " * rhs_p * "\n"

           lhs_m = "$(y[(2*n2+1)]), $(y[(2*n2+2)])" * foldl(*, vmap(i -> ", $(y[(2*(i+n2)-1)]), $(y[(2*(i+n2))])", 2:n2))
           rhs_m = "$(sat_expr("-", "$(t[1])", "$(t[1+n2])", d[n2]))" * foldl(*, vmap(i -> ", $(sat_expr("tmp$(i-3+n2)", d[i+n2-1]))", 2:n2))
           s3m = lhs_m * " = " * rhs_m * "\n"
        end
      end
    else
      if isnothing(w)
        if root
          lhs_p = "$(y[1]), $(y[2])" * foldl(*, vmap(i -> ", $(y[(2*i-1)]), $(y[(2*i)])", 2:n2))
          rhs_p = "$(t[1])_r + $(t[1+n2])_r, $(t[1])_i + $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r + $(t[i+n2])_r, $(t[i])_i + $(t[i+n2])_i", 2:n2))
          s3p = "$py" * "\n" * lhs_p * " = " * rhs_p * "\n"

          lhs_m = "$(y[(2*n2+1)]), $(y[(2*n2+2)])" * foldl(*, vmap(i -> ", $(y[(2*(i+n2)-1)]), $(y[(2*(i+n2))])", 2:n2))
          rhs_m = "$(t[1])_r - $(t[1+n2])_r, $(t[1])_i - $(t[1+n2])_i" * foldl(*, vmap(i -> ", $(t[i])_r - $(t[i+n2])_r, $(t[i])_i - $(t[i+n2])_i", 2:n2))
          s3m = lhs_m * " = " * rhs_m * "\n"
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
  s = n == 4 ? load_real_imag_gen(x; mode=mode, T=T, input=input_buffer) * "\n" * s1 * s2 * s3p * s3m : s1 * s2 * s3p * s3m
  return s
end

# Complete SIMD FFT kernel generator
"""
recfft2_simd(y, x, d, w, root, ::Type{T}; SIMD_WIDTH=256, tmp_base=1, mode=:default, py="")

A SIMD-aware code-generator replacement for your scalar recfft2 codegen.
Generates horizontal SIMD operations like vfft8_fastest.

# Arguments
- `y, x`: Output and input symbolic variable arrays
- `d, w`: Twiddle factors (d for DIT, w for general)
- `root`: Whether this is the root call
- `T`: Float type (Float32, Float64, Float16)
- `SIMD_WIDTH`: Hardware SIMD width in bits (256 for AVX2, 512 for AVX512)
- `tmp_base`: Base index for temporary variables
- `mode`: Load/store mode (:default, :vgather, etc.)
- `py`: 

# Strategy
- Packs multiple complex numbers horizontally in Vec{N,T}
- Uses shufflevector for data rearrangement
- Uses signflip (XOR) for sign corrections
- Ensures Vec{N,T} fits in hardware SIMD registers
"""
function recfft2_simd(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base=1, mode=:vgather, py="", complexes_per_vec=4, input_buffer="x", output_buffer="y") where T <: AbstractFloat
    n = length(x)  # Number of complex numbers

    SIMD_WIDTH = floats_per_vec * sizeof(T) * 8

    # Helper: generate shuffle indices to extract real or imag parts
    function make_extract_indices(which::Symbol, n_complex::Int)
        # which = :real extracts [0, 2, 4, 6, ...] (0-indexed)
        # which = :imag extracts [1, 3, 5, 7, ...]
        offset = (which == :real) ? 0 : 1
        return join([offset + 2*i for i in 0:n_complex-1], ", ")
    end

    # Helper: generate shuffle indices to interleave two vectors
    function make_interleave_indices(n_complex::Int)
        # Interleave v_r and v_i: [r0, i0, r1, i1, ...]
        # v_r is indices 0..(n_complex-1), v_i is indices n_complex..(2*n_complex-1)
        indices = Int[]
        for i in 0:n_complex-1
            push!(indices, i)              # from v_r
            push!(indices, i + n_complex)  # from v_i
        end
        return join(indices, ", ")
    end

    # Helper: generate signflip mask
    function make_signflip_mask(pattern::Vector{Bool}, vec_size::Int, ::Type{T})
        # pattern[i] = true means flip sign at position i
        # For Float32: sign bit is 0x80000000, for Float64: 0x8000000000000000
        UIntType = T == Float32 ? "UInt32" : (T == Float64 ? "UInt64" : "UInt16")
        sign_bit = T == Float32 ? "0x80000000" : (T == Float64 ? "0x8000000000000000" : "0x8000")
        zero_bit = T == Float32 ? "0x00000000" : (T == Float64 ? "0x0000000000000000" : "0x0000")

        mask_vals = [pattern[min(i, length(pattern))] ? sign_bit : zero_bit for i in 1:vec_size]
        return "Vec{$vec_size,$UIntType}(($(join(mask_vals, ", "))))"
    end

    if n == 1
        return ""

    elseif n == 2
        # BASE CASE: Radix-2 butterfly
        # Scalar equivalent: y1 = x1 + x2, y2 = x1 - x2 (with twiddles)
        # SIMD: Load each complex number as Vec{2,T}, operate on them

        s = if !isnothing(d)
            if isnothing(w)
                if root
                    # Extract indices from variable names for loads (e.g., "v1" -> 1, "v3" -> 3)
                    idx1 = parse(Int, match(r"(\d+)", x[1]).captures[1])
                    idx2 = parse(Int, match(r"(\d+)", x[2]).captures[1])
                    pos1 = 2 * (idx1 - 1) + 1  # Float position (1-indexed)
                    pos2 = 2 * (idx2 - 1) + 1

                    # Extract store positions from y array (first complex: y[1],y[2], second: y[3],y[4])
                    store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
                    store_pos2 = parse(Int, match(r"\[(\d+)\]", y[3]).captures[1])

                    # Load, butterfly, apply twiddles, store
                    """
                    # Load 2 complex numbers as Vec{2,T} each
                    $(x[1]) = vload(Vec{2,$T}, $input_buffer, $pos1)
                    $(x[2]) = vload(Vec{2,$T}, $input_buffer, $pos2)

                    # Butterfly
                    tmp0 = $(x[1]) - $(x[2])
                    $py
                    t1 = $(x[1]) + $(x[2])
                    t2 = $(sat_expr_simd("tmp0", d[1], T, 2))

                    # Store
                    vstore(t1, $output_buffer, $store_pos1)
                    vstore(t2, $output_buffer, $store_pos2)
                    """
                end
            end
        else
            if root
                # Extract indices from variable names for loads
                idx1 = parse(Int, match(r"(\d+)", x[1]).captures[1])
                idx2 = parse(Int, match(r"(\d+)", x[2]).captures[1])
                pos1 = 2 * (idx1 - 1) + 1
                pos2 = 2 * (idx2 - 1) + 1

                # Extract store positions from y array (first complex: y[1],y[2], second: y[3],y[4])
                store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
                store_pos2 = parse(Int, match(r"\[(\d+)\]", y[3]).captures[1])

                """
                # Load
                $(x[1]) = vload(Vec{2,$T}, $input_buffer, $pos1)
                $(x[2]) = vload(Vec{2,$T}, $input_buffer, $pos2)

                # Butterfly
                t1 = $(x[1]) + $(x[2])
                t2 = $(x[1]) - $(x[2])

                # Store
                vstore(t1, $output_buffer, $store_pos1)
                vstore(t2, $output_buffer, $store_pos2)
                """
            else
                if isnothing(w)
                    # Non-root, no twiddles: simple butterfly
                    """
                    $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(x[1]) - $(x[2])
                    """
                else
                    # Non-root with twiddles
                    if w[1] == "1"
                        """
                        $(y[1]) = $(x[1]) + $(x[2])
                        $(y[2]) = $(sat_expr_simd("($(x[1]) - $(x[2]))", w[2], T, 2))
                        """
                    else
                        """
                        $(y[1]) = $(sat_expr_simd("($(x[1]) + $(x[2]))", w[1], T, 2))
                        $(y[2]) = $(sat_expr_simd("($(x[1]) - $(x[2]))", w[2], T, 2))
                        """
                    end
                end
            end
        end

        return something(s, "")

    else
        # RECURSIVE CASE: Split into two sub-transforms, then combine
        n2 = n ÷ 2
        t = ["tvec$(tmp_base + i - 1)" for i in 1:n]
        new_tmp_base = tmp_base + n

        # Generate loads if this is the root call
        load_code, store_setup = if root
            # Use load_gen_simd to load all input variables
            raw_load = load_gen_simd(x; mode=mode, T=T, ptr_name=input_buffer, SIMD_BITS=SIMD_WIDTH)

            # Map loaded variables to expected names using shufflevector
            # load_gen_simd creates either: l_all, or v1, v2, v3, ...
            # We need to extract specific complexes to x[1], x[2], etc.
            assignments = String[]
            if n <= complexes_per_vec
                # Single vector loaded as l_all
                # Extract each complex number (2 floats) to individual variables
                for (i, var) in enumerate(x)
                    # Calculate indices for i-th complex number (0-indexed for shufflevector)
                    idx_start = 2 * (i - 1)
                    shuffle_indices = join([idx_start, idx_start + 1], ", ")
                    push!(assignments, "$var = shufflevector(l_all, Val(($shuffle_indices)))")
                end
            else
                # Multiple vectors loaded as v1, v2, v3, ...
                # Need to extract and potentially combine across vectors
                n_vecs = cld(n, complexes_per_vec)  # Number of loaded vectors

                for (i, var) in enumerate(x)
                    # Which complex number is this (1-indexed)
                    complex_idx = i
                    # Which vector contains it (1-indexed)
                    vec_idx = ((complex_idx - 1) ÷ complexes_per_vec) + 1
                    # Position within that vector (0-indexed for shufflevector)
                    pos_in_vec = (complex_idx - 1) % complexes_per_vec
                    # Float indices within the vector (0-indexed)
                    idx_start = 2 * pos_in_vec
                    shuffle_indices = join([idx_start, idx_start + 1], ", ")

                    push!(assignments, "$var = shufflevector(l$vec_idx, Val(($shuffle_indices)))")
                end
            end

            load_str = raw_load * "\n" * join(assignments, "\n") * "\n"
            #store_str = "py = reinterpret($T, y)\n"
            store_str = "\n"
            (load_str, store_str)
        else
            ("", "")
        end

        # Recursively generate sub-transforms
        # Even indices: x[1], x[3], x[5], ...
        # Odd indices: x[2], x[4], x[6], ...
        s1 = recfft2_simd(t[1:n2], x[1:2:n], nothing, nothing, false, T,
                          floats_per_vec ÷ 2, new_tmp_base, mode, py, complexes_per_vec, input_buffer, output_buffer)

        # Second half with twiddles (always compute for proper FFT, like scalar recfft2)
        twiddles_second_half = get_twiddle_expression(collect(0:n2-1), n; T=T, accuracy=nothing)
        s2 = recfft2_simd(t[n2+1:n], x[2:2:n], nothing, twiddles_second_half, false, T,
                          floats_per_vec ÷ 2, new_tmp_base, mode, py, complexes_per_vec, input_buffer, output_buffer)

        # Generate temporary declarations for butterfly combinations
        tmp_decls = if n > 2
            parts = String[]
            # Create sum and diff temporaries
            for i in 2:n2
                push!(parts, "tmp$(i-2) = $(t[i]) + $(t[i+n2])")
            end
            for i in 2:n2
                push!(parts, "tmp$(n2+i-3) = $(t[i]) - $(t[i+n2])")
            end
            join(parts, "\n") * "\n"
        else
            ""
        end

        # The tmp variables and t variables from sub-recursions are ALWAYS Vec{2, T}
        # because each recursive call outputs individual complex numbers as Vec{2, T},
        # regardless of the floats_per_vec parameter passed to the sub-call.
        tmp_vec_size = 2

        # Final butterfly layer
        s3p, s3m = "", ""

        if !isnothing(d)
            # Using D-matrix twiddles (DIT)
            if isnothing(w)
                if root
                    # Create intermediate Vec variables
                    vec_vars_p = ["v_out$(i)" for i in 1:n2]
                    vec_vars_m = ["v_out$(i+n2)" for i in 1:n2]

                    # First half: t[i] + t[i+n2] with twiddles
                    lhs_p = join(vec_vars_p, ", ")
                    rhs_parts_p = ["$(t[1]) + $(t[1+n2])"]
                    for i in 2:n2
                        push!(rhs_parts_p, "$(sat_expr_simd("tmp$(i-2)", d[i-1], T, tmp_vec_size))")
                    end
                    rhs_p = join(rhs_parts_p, ", ")

                    # Second half: t[i] - t[i+n2] with twiddles
                    lhs_m = join(vec_vars_m, ", ")
                    rhs_parts_m = ["$(sat_expr_simd("($(t[1]) - $(t[1+n2]))", d[n2], T, tmp_vec_size))"]
                    for i in 2:n2
                        push!(rhs_parts_m, "$(sat_expr_simd("tmp$(n2+i-3)", d[n2+i-1], T, tmp_vec_size))")
                    end
                    rhs_m = join(rhs_parts_m, ", ")

                    s3p = "$store_setup$py\n$tmp_decls$lhs_p = $rhs_p\n"
                    s3m = "$lhs_m = $rhs_m\n"

                    # Combine Vec{2,T} results into wide vectors and store
                    # Each vec_vars element is Vec{2,T}, combine n2 of them into Vec{floats_per_vec,T}
                    if n2 * 2 == floats_per_vec && n2 == 4
                        # FFT8 case: 4 Vec{2,T} -> Vec{8,T}
                        # shufflevector only takes 1-2 sources, so do nested combines
                        # Use let block to avoid register pressure
                        s3p *= "let\n"
                        s3p *= "    y_tmp1 = shufflevector($(vec_vars_p[1]), $(vec_vars_p[2]), Val((0, 1, 2, 3)))\n"
                        s3p *= "    y_tmp2 = shufflevector($(vec_vars_p[3]), $(vec_vars_p[4]), Val((0, 1, 2, 3)))\n"
                        s3p *= "    y_wide1 = shufflevector(y_tmp1, y_tmp2, Val((0, 1, 2, 3, 4, 5, 6, 7)))\n"
                        s3p *= "    vstore(y_wide1, $output_buffer, 1)\n"
                        s3p *= "end\n"
                        s3m *= "let\n"
                        s3m *= "    y_tmp3 = shufflevector($(vec_vars_m[1]), $(vec_vars_m[2]), Val((0, 1, 2, 3)))\n"
                        s3m *= "    y_tmp4 = shufflevector($(vec_vars_m[3]), $(vec_vars_m[4]), Val((0, 1, 2, 3)))\n"
                        s3m *= "    y_wide2 = shufflevector(y_tmp3, y_tmp4, Val((0, 1, 2, 3, 4, 5, 6, 7)))\n"
                        s3m *= "    vstore(y_wide2, $output_buffer, $(floats_per_vec + 1))\n"
                        s3m *= "end\n"
                    else
                        # Fall back to individual stores - extract positions from y array
                        # For each Vec{2,T} result, extract the starting position from the y template
                        store_positions_p = [parse(Int, match(r"\[(\d+)\]", y[2*i - 1]).captures[1]) for i in 1:n2]
                        store_positions_m = [parse(Int, match(r"\[(\d+)\]", y[2*(i+n2) - 1]).captures[1]) for i in 1:n2]

                        stores_p = join(["vstore($(vec_vars_p[i]), $output_buffer, $(store_positions_p[i]))" for i in 1:n2], "\n")
                        stores_m = join(["vstore($(vec_vars_m[i]), $output_buffer, $(store_positions_m[i]))" for i in 1:n2], "\n")
                        s3p *= "$stores_p\n"
                        s3m *= "$stores_m\n"
                    end
                end
            end
        else
            if isnothing(w)
                # No twiddles: simple butterfly
                if root
                    # Create intermediate Vec variables
                    vec_vars_p = ["v_out$(i)" for i in 1:n2]
                    vec_vars_m = ["v_out$(i+n2)" for i in 1:n2]

                    lhs_p = join(vec_vars_p, ", ")
                    rhs_p = join(["$(t[i]) + $(t[i+n2])" for i in 1:n2], ", ")

                    lhs_m = join(vec_vars_m, ", ")
                    rhs_m = join(["$(t[i]) - $(t[i+n2])" for i in 1:n2], ", ")

                    s3p = "$store_setup$py\n$lhs_p = $rhs_p\n"
                    s3m = "$lhs_m = $rhs_m\n"

                    # Combine Vec{2,T} results into wide vectors and store
                    if n2 * 2 == floats_per_vec && n2 == 4
                        # FFT8 case: 4 Vec{2,T} -> Vec{8,T}
                        # shufflevector only takes 1-2 sources, so do nested combines
                        # Use let block to avoid register pressure
                        s3p *= "let\n"
                        s3p *= "    y_tmp1 = shufflevector($(vec_vars_p[1]), $(vec_vars_p[2]), Val((0, 1, 2, 3)))\n"
                        s3p *= "    y_tmp2 = shufflevector($(vec_vars_p[3]), $(vec_vars_p[4]), Val((0, 1, 2, 3)))\n"
                        s3p *= "    y_wide1 = shufflevector(y_tmp1, y_tmp2, Val((0, 1, 2, 3, 4, 5, 6, 7)))\n"
                        s3p *= "    vstore(y_wide1, $output_buffer, 1)\n"
                        s3p *= "end\n"
                        s3m *= "let\n"
                        s3m *= "    y_tmp3 = shufflevector($(vec_vars_m[1]), $(vec_vars_m[2]), Val((0, 1, 2, 3)))\n"
                        s3m *= "    y_tmp4 = shufflevector($(vec_vars_m[3]), $(vec_vars_m[4]), Val((0, 1, 2, 3)))\n"
                        s3m *= "    y_wide2 = shufflevector(y_tmp3, y_tmp4, Val((0, 1, 2, 3, 4, 5, 6, 7)))\n"
                        s3m *= "    vstore(y_wide2, $output_buffer, $(floats_per_vec + 1))\n"
                        s3m *= "end\n"
                    else
                        # Fall back to individual stores - extract positions from y array
                        # For each Vec{2,T} result, extract the starting position from the y template
                        store_positions_p = [parse(Int, match(r"\[(\d+)\]", y[2*i - 1]).captures[1]) for i in 1:n2]
                        store_positions_m = [parse(Int, match(r"\[(\d+)\]", y[2*(i+n2) - 1]).captures[1]) for i in 1:n2]

                        stores_p = join(["vstore($(vec_vars_p[i]), $output_buffer, $(store_positions_p[i]))" for i in 1:n2], "\n")
                        stores_m = join(["vstore($(vec_vars_m[i]), $output_buffer, $(store_positions_m[i]))" for i in 1:n2], "\n")
                        s3p *= "$stores_p\n"
                        s3m *= "$stores_m\n"
                    end
                else
                    lhs_p = join([y[i] for i in 1:n2], ", ")
                    rhs_p = join(["$(t[i]) + $(t[i+n2])" for i in 1:n2], ", ")

                    lhs_m = join([y[i+n2] for i in 1:n2], ", ")
                    rhs_m = join(["$(t[i]) - $(t[i+n2])" for i in 1:n2], ", ")

                    s3p = "$lhs_p = $rhs_p\n"
                    s3m = "$lhs_m = $rhs_m\n"
                end
            else
                # With twiddles
                s3p = tmp_decls

                lhs_p = join([y[i] for i in 1:n2], ", ")
                rhs_parts_p = [w[1] == "1" ? "$(t[1]) + $(t[1+n2])" : "$(sat_expr_simd("($(t[1]) + $(t[1+n2]))", w[1], T, tmp_vec_size))"]
                for i in 2:n2
                    push!(rhs_parts_p, "$(sat_expr_simd("tmp$(i-2)", w[i], T, tmp_vec_size))")
                end
                rhs_p = join(rhs_parts_p, ", ")

                lhs_m = join([y[i+n2] for i in 1:n2], ", ")
                rhs_parts_m = ["$(sat_expr_simd("($(t[1]) - $(t[1+n2]))", w[n2+1], T, tmp_vec_size))"]
                for i in 2:n2
                    push!(rhs_parts_m, "$(sat_expr_simd("tmp$(n2+i-3)", w[n2+i], T, tmp_vec_size))")
                end
                rhs_m = join(rhs_parts_m, ", ")

                s3p *= "$lhs_p = $rhs_p\n"
                s3m = "$lhs_m = $rhs_m\n"
            end
        end

        return load_code * s1 * s2 * s3p * s3m
    end
end

function vec_mul_twiddle_d(vec_name::String, c_idx::Int, s_idx::Int, ::Type{T}, floats_per_vec::Int) where T
    # Use d[c_idx] for cos, d[s_idx] for sin in vectorized complex multiplication
    c = "d[$c_idx]"
    s = "d[$s_idx]"
    # Call the same CISPI multiplication code, but with direct references to d
    return vec_mul_cispi_code(vec_name, c, s, false, T, floats_per_vec)  # false = Q4 quadrant (cos - i*sin)
end

# SIMD code generation helpers for complex twiddle multiplication
# These generate actual SIMD.jl expressions that will be inlined
function sat_expr_simd(vec_name::String, twiddle, ::Type{T}, floats_per_vec::Int) where T
    if twiddle isa String
        if twiddle == "1"
            return vec_name
        elseif twiddle == "-1"
            # Negate all components: -1*(r+ii) = -r - ii
            return "-$vec_name"
        elseif twiddle == "im"
            # i*(r+ii) = ir - i = -i + ir → [r,i] becomes [-i, r]
            # Generate shuffle indices to swap r,i pairs: [r1,i1,r2,i2,...] -> [i1,r1,i2,r2,...]
            swap_indices = Int[]
            for i in 1:2:floats_per_vec
                push!(swap_indices, i+1)  # imag component (1-based)
                push!(swap_indices, i)    # real component (1-based)
            end
            # Generate sign mask to negate real parts (now at even indices after swap)
            sign_mask = join(["0x$(i % 2 == 1 ? "80000000" : "00000000")" for i in 1:floats_per_vec], ", ")
            return "shufflevector(signflip($vec_name, Vec{$floats_per_vec,UInt32}(($sign_mask))), Val(($(join(swap_indices .- 1, ", ")))))"
        elseif twiddle == "-im"
            # -i*(r+ii) = -ir + i = i - ir → [r,i] becomes [i, -r]
            # Step 1: signflip to negate real parts: [r,i] -> [-r,i]
            # Step 2: swap to get [i,-r]
            swap_indices = Int[]
            for i in 1:2:floats_per_vec
                push!(swap_indices, i+1)  # imag component (1-based)
                push!(swap_indices, i)    # real component (1-based)
            end
            # Generate sign mask to negate real parts (at even indices, i % 2 == 1 in 1-based)
            sign_mask = join(["0x$(i % 2 == 1 ? "80000000" : "00000000")" for i in 1:floats_per_vec], ", ")
            return "shufflevector(signflip($vec_name, Vec{$floats_per_vec,UInt32}(($sign_mask))), Val(($(join(swap_indices .- 1, ", ")))))"
        elseif twiddle == "INV_SQRT2_Q4"
            # (r+ii) * (1-i)/√2 = (r+i)/√2 + i(i-r)/√2 = ((r+i), (i-r))/√2

            if floats_per_vec == 2
                # Special case for Vec{2, Float32} - single complex number [r, i]
                # Want [(r+i)/√2, (i-r)/√2]
                return """(let
                    v = $vec_name
                    v_swap = shufflevector(v, Val((1, 0)))
                    v_sum = v + v_swap
                    v_diff = v - v_swap
                    shufflevector(v_sum, v_diff, Val((0, 3))) * INV_SQRT2
                end)"""
            else
                num_complex = floats_per_vec ÷ 2
                # Extract real and imag parts
                r_indices = [2i-1 for i in 1:num_complex]
                i_indices = [2i for i in 1:num_complex]

                # Interleave for final result: r1, i1, r2, i2, ...
                interleave_indices = Int[]
                for i in 0:num_complex-1
                    push!(interleave_indices, i)
                    push!(interleave_indices, i + num_complex)
                end

                return """(let
                    v_r = shufflevector($(vec_name), Val(($(join(r_indices .- 1, ", ")))))
                    v_i = shufflevector($(vec_name), Val(($(join(i_indices .- 1, ", ")))))
                    v_sum = v_r + v_i
                    v_diff = v_i - v_r
                    shufflevector(v_sum, v_diff, Val(($(join(interleave_indices, ", "))))) * INV_SQRT2
                end)"""
            end
        elseif twiddle == "-INV_SQRT2_Q1"
            # -(r+ii) * (1+i)/√2 = [-(r-i)/√2, -(r+i)/√2] = [(i-r)/√2, -(r+i)/√2]

            if floats_per_vec == 2
                # Special case for Vec{2, Float32} - single complex number [r, i]
                # Want [(i-r)/√2, -(r+i)/√2]
                return """(let
                    v_swap = shufflevector($vec_name, Val((1, 0)))
                    v_sum = $(vec_name) + v_swap
                    v_diff = $(vec_name) - v_swap
                    shufflevector(v_diff, -v_sum, Val((1, 2))) * INV_SQRT2
                end)"""
            else
                num_complex = floats_per_vec ÷ 2
                r_indices = [2i-1 for i in 1:num_complex]
                i_indices = [2i for i in 1:num_complex]

                # Interleave for final result: r1, i1, r2, i2, ...
                interleave_indices = Int[]
                for i in 0:num_complex-1
                    push!(interleave_indices, i)
                    push!(interleave_indices, i + num_complex)
                end

                return """(let
                    v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                    v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                    v_sum = v_r + v_i
                    v_diff = v_i - v_r
                    shufflevector(v_diff, -v_sum, Val(($(join(interleave_indices, ", "))))) * INV_SQRT2
                end)"""
            end
        end
            try
              # Meta.parse returns an expression, which needs to be evaluated/used
              # or passed to the Tuple logic. We'll evaluate it to get the value.
              twiddle_parsed = Core.eval(Main, Meta.parse(twiddle))
                  
              # If successfully parsed into a Tuple, jump to the Tuple logic below
              if twiddle_parsed isa Tuple
                  return sat_expr_simd(vec_name, twiddle_parsed, T, floats_per_vec)
              end
            catch
              # If parsing failed, it's a genuinely unknown string twiddle factor.
                error("Unknown string twiddle factor: $twiddle")
            end

            error("Unknown string twiddle factor: $twiddle")
    elseif twiddle isa Tuple
        # General CISPI twiddle
        num, den, quadrant = twiddle
        c = "COSPI_$(num)_$(den)"
        s = "SINPI_$(num)_$(den)"

        #return vec_mul_cispi_code(vec_name, c, s, parsed.q1, T, floats_per_vec)

        num_complex = floats_per_vec ÷ 2
        r_indices = [2i-1 for i in 1:num_complex]
        i_indices = [2i for i in 1:num_complex]

        # Interleave for final result: r1, i1, r2, i2, ...
        interleave_indices = Int[]
        for i in 0:num_complex-1
            push!(interleave_indices, i)            # from out_r
            push!(interleave_indices, i + num_complex)  # from out_i
        end

        # Complex multiply using muladd for FMA optimization
        # cispi(θ) = cos(πθ) + i*sin(πθ)
        if quadrant == Q1
            # Q1: cispi(θ) = cos + i*sin
            # (r+ii)*(c+si) = (rc-si) + i(rs+ic)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(v_r, $c, -v_i * $s)
                out_i = muladd(v_r, $s, v_i * $c)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == Q4
            # Q4: cispi(-θ) = cos - i*sin
            # (r+ii)*(c-si) = (rc+si) + i(-rs+ic)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(v_r, $c, v_i * $s)
                out_i = muladd(-v_r, $s, v_i * $c)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == NegQ1
            # NegQ1: -cispi(θ) = -cos - i*sin
            # (r+ii)*(-c-si) = (-rc+si) + i(-rs-ic)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(-v_r, $c, v_i * $s)
                out_i = muladd(-v_r, $s, -v_i * $c)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == NegQ4
            # NegQ4: -cispi(-θ) = -cos + i*sin
            # (r+ii)*(-c+si) = (-rc-si) + i(rs-ic)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(-v_r, $c, -v_i * $s)
                out_i = muladd(v_r, $s, -v_i * $c)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == ImQ1
            # ImQ1: i*cispi(θ) = -sin + i*cos
            # (r+ii)*(-s+ci) = (-rs-ic) + i(rc-is)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(-v_r, $s, -v_i * $c)
                out_i = muladd(v_r, $c, -v_i * $s)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == ImQ4
            # ImQ4: i*cispi(-θ) = sin + i*cos
            # (r+ii)*(s+ci) = (rs-ic) + i(rc+is)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(v_r, $s, -v_i * $c)
                out_i = muladd(v_r, $c, v_i * $s)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == NegImQ1
            # NegImQ1: -i*cispi(θ) = sin - i*cos
            # (r+ii)*(s-ci) = (rs+ic) + i(-rc+is)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(v_r, $s, v_i * $c)
                out_i = muladd(-v_r, $c, v_i * $s)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        elseif quadrant == NegImQ4
            # NegImQ4: -i*cispi(-θ) = -sin - i*cos
            # (r+ii)*(-s-ci) = (-rs+ic) + i(-rc-is)
            """(let
                v_r = shufflevector($vec_name, Val(($(join(r_indices .- 1, ", ")))))
                v_i = shufflevector($vec_name, Val(($(join(i_indices .- 1, ", ")))))
                out_r = muladd(-v_r, $s, v_i * $c)
                out_i = muladd(-v_r, $c, -v_i * $s)
                shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
            end)"""
        else
            error("Unknown quadrant: $quadrant")
        end
    end
end

# Wraper: apply twiddle to (vec1 op vec2)
function vec_mul_twiddle_op(op::String, vec1::String, vec2::String, twiddle::String, ::Type{T}, floats_per_vec::Int) where T
    if twiddle == "1"
        return "$vec1 $op $vec2"
    else
        return vec_mul_twiddle("($vec1 $op $vec2)", twiddle, T, floats_per_vec)
    end
end

# Vectorized XOR operation to flip sign at chosen vector elements

@inline function vfft8(x::Vector{Float32}, y::Vector{Float32}) 
    @inbounds @fastmath begin
        INV_SQRT2 = 0.7071067811865476f0
        
        # Load all 16 floats in just 2 aligned loads!
        LANE = VecRange{8}(0)
        ymm_low = x[LANE + 1]   # x[1:8]   = [x1_r, x1_i, x2_r, x2_i, x3_r, x3_i, x4_r, x4_i]
        ymm_high = x[LANE + 9]  # x[9:16]  = [x5_r, x5_i, x6_r, x6_i, x7_r, x7_i, x8_r, x8_i]
        
        # Extract needed components with shuffles
        # First radix-4: need [x1_r, x1_i, x3_r, x3_i] and [x5_r, x5_i, x7_r, x7_i]
        v1_a = shufflevector(ymm_low, Val((0, 1, 4, 5)))   # [x1_r, x1_i, x3_r, x3_i]
        v2_a = shufflevector(ymm_high, Val((0, 1, 4, 5)))  # [x5_r, x5_i, x7_r, x7_i]
        
        # Second radix-4: need [x2_r, x2_i, x4_r, x4_i] and [x6_r, x6_i, x8_r, x8_i]
        v1_b = shufflevector(ymm_low, Val((2, 3, 6, 7)))   # [x2_r, x2_i, x4_r, x4_i]
        v2_b = shufflevector(ymm_high, Val((2, 3, 6, 7)))  # [x6_r, x6_i, x8_r, x8_i]
        
        # ... rest of FFT computation (same as before) ...
        # First radix-4
        t9_11 = v1_a + v2_a
        t10_12_raw = v1_a - v2_a
        
        # Fix t12: [t10_r, t10_i, -t12_i, t12_r] -> [t10_r, t10_i, t12_r, t12_i]
        sign_mask = Vec{4,UInt32}((0x00000000, 0x00000000, 0x80000000, 0x00000000))
        t10_12 = shufflevector(signflip(t10_12_raw, sign_mask), Val((0, 1, 3, 2)))
        
        h1_a = shufflevector(t9_11, t10_12, Val((0, 1, 4, 5)))
        h2_a = shufflevector(t9_11, t10_12, Val((2, 3, 6, 7)))
        
        t1_2 = h1_a + h2_a
        t3_4 = h1_a - h2_a
        ymm1 = shufflevector(t1_2, t3_4, Val((0,1,2,3,4,5,6,7)))
        
        # Second radix-4
        t9_11_b = v1_b + v2_b
        t10_12_raw_b = v1_b - v2_b
        t10_12_b = shufflevector(signflip(t10_12_raw_b, sign_mask), Val((0, 1, 3, 2)))
        
        h1_b = shufflevector(t9_11_b, t10_12_b, Val((0, 1, 4, 5)))
        h2_b = shufflevector(t9_11_b, t10_12_b, Val((2, 3, 6, 7)))
        
        t5 = h1_b + h2_b
        tmp1_pre = h1_b - h2_b
        
        # Twiddle computation (simplified with FMA-style patterns)
        tmp0 = shufflevector(t5, Val((2, 3, 2, 3)))
        tmp0_swapped = shufflevector(tmp0, Val((1, 0, 1, 0)))
        t6 = (tmp0 + tmp0_swapped * Vec{4,Float32}((1.0f0, -1.0f0, 0.0f0, 0.0f0))) * 
             Vec{4,Float32}((INV_SQRT2, INV_SQRT2, 0.0f0, 0.0f0))
        
        t7_raw = shufflevector(tmp1_pre, Val((1, 0, 1, 0)))
        sign_mask_t7 = Vec{4,UInt32}((0x00000000, 0x80000000, 0x00000000, 0x00000000))
        t7 = signflip(t7_raw, sign_mask_t7)
        
        tmp1 = shufflevector(tmp1_pre, Val((2, 3, 2, 3)))
        tmp1_swapped = shufflevector(tmp1, Val((1, 0, 1, 0)))
        tmp1_sum = tmp1 + tmp1_swapped
        tmp1_diff = tmp1_swapped - tmp1
        t8_pre = shufflevector(tmp1_diff, tmp1_sum, Val((0, 5, 0, 0)))
        sign_mask_t8 = Vec{4,UInt32}((0x00000000, 0x80000000, 0x00000000, 0x00000000))
        t8 = signflip(t8_pre, sign_mask_t8) * Vec{4,Float32}((INV_SQRT2, INV_SQRT2, 0.0f0, 0.0f0))
        
        t5_6 = shufflevector(t5, t6, Val((0,1,4,5)))
        t7_8 = shufflevector(t7, t8, Val((0,1,4,5)))
        ymm2 = shufflevector(t5_6, t7_8, Val((0,1,2,3,4,5,6,7)))
        
        # Final butterfly
        y1_4 = ymm1 + ymm2
        y5_8 = ymm1 - ymm2
        
        vstore(y1_4, y, 1)
        vstore(y5_8, y, 9)
    end
end

@inline function vfft8_fastest(x::Vector{Float32}, y::Vector{Float32})
    @inbounds @fastmath begin
        INV_SQRT2 = 0.7071067811865476f0

        # ========== ULTRA-FAST LOAD: Only 2 instructions! ==========
        LANE = VecRange{8}(0)
        ymm_all = x[LANE + 1]  # Load all 16 floats in ONE vector!

        # Split into two 128-bit halves
        xmm_low = shufflevector(ymm_all, Val((0, 1, 2, 3)))   # x[1:4]
        xmm_high = shufflevector(ymm_all, Val((4, 5, 6, 7)))  # x[5:8]

        # Load second half
        ymm_all2 = x[LANE + 9]
        xmm_low2 = shufflevector(ymm_all2, Val((0, 1, 2, 3)))   # x[9:12]
        xmm_high2 = shufflevector(ymm_all2, Val((4, 5, 6, 7)))  # x[13:16]

        # Now continue with the same butterfly structure as before
        # (using xmm_low, xmm_high, xmm_low2, xmm_high2 as xmm0, xmm1, xmm2, xmm3)

        # ========== FIRST RADIX-4 ==========
        v1_a = shufflevector(xmm_low, xmm_high, Val((0, 1, 4, 5)))
        v2_a = shufflevector(xmm_low2, xmm_high2, Val((0, 1, 4, 5)))

        t9_11_a = v1_a + v2_a
        diff_a = v1_a - v2_a
        diff_a_swapped = shufflevector(diff_a, Val((0, 1, 3, 2)))
        sign_mask_t12 = Vec{4,UInt32}((0x00000000, 0x00000000, 0x00000000, 0x80000000))
        t10_12_a = signflip(diff_a_swapped, sign_mask_t12)

        h1_a = shufflevector(t9_11_a, t10_12_a, Val((0, 1, 4, 5)))
        h2_a = shufflevector(t9_11_a, t10_12_a, Val((2, 3, 6, 7)))

        t1_2 = h1_a + h2_a
        t3_4 = h1_a - h2_a

        # ========== SECOND RADIX-4 ==========
        v1_b = shufflevector(xmm_low, xmm_high, Val((2, 3, 6, 7)))
        v2_b = shufflevector(xmm_low2, xmm_high2, Val((2, 3, 6, 7)))

        t9_11_b = v1_b + v2_b
        diff_b = v1_b - v2_b
        diff_b_swapped = shufflevector(diff_b, Val((0, 1, 3, 2)))
        t10_12_b = signflip(diff_b_swapped, sign_mask_t12)

        h1_b = shufflevector(t9_11_b, t10_12_b, Val((0, 1, 4, 5)))
        h2_b = shufflevector(t9_11_b, t10_12_b, Val((2, 3, 6, 7)))

        t5_tmp0 = h1_b + h2_b
        diff_t9t11_tmp1 = h1_b - h2_b

        # ========== TWIDDLES ==========
        tmp0 = shufflevector(t5_tmp0, t5_tmp0, Val((2, 3, 2, 3)))
        tmp0_rotated = shufflevector(tmp0, Val((1, 0, 1, 0)))
        sign_mask_rot = Vec{4,UInt32}((0x00000000, 0x80000000, 0x00000000, 0x00000000))
        tmp0_rotated_signed = signflip(tmp0_rotated, sign_mask_rot)
        t6_unscaled = tmp0 + tmp0_rotated_signed
        t6 = t6_unscaled * Vec{4,Float32}((INV_SQRT2, INV_SQRT2, 0.0f0, 0.0f0))

        t7_raw = shufflevector(diff_t9t11_tmp1, Val((1, 0, 1, 0)))
        t7 = signflip(t7_raw, sign_mask_rot)

        tmp1 = shufflevector(diff_t9t11_tmp1, diff_t9t11_tmp1, Val((2, 3, 2, 3)))
        tmp1_swapped = shufflevector(tmp1, Val((1, 0, 1, 0)))
        tmp1_sum = tmp1 + tmp1_swapped
        tmp1_diff = tmp1_swapped - tmp1
        t8_unsigned = shufflevector(tmp1_diff, tmp1_sum, Val((0, 5, 0, 0)))
        t8_signed = signflip(t8_unsigned, sign_mask_rot)
        t8 = t8_signed * Vec{4,Float32}((INV_SQRT2, INV_SQRT2, 0.0f0, 0.0f0))

        t5_6 = shufflevector(t5_tmp0, t6, Val((0, 1, 4, 5)))
        t7_8 = shufflevector(t7, t8, Val((0, 1, 4, 5)))

        # ========== FINAL BUTTERFLY ==========
        ymm1 = shufflevector(t1_2, t3_4, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        ymm2 = shufflevector(t5_6, t7_8, Val((0, 1, 2, 3, 4, 5, 6, 7)))

        y1_8_top = ymm1 + ymm2
        y1_8_bot = ymm1 - ymm2

        vstore(y1_8_top, y, 1)
        vstore(y1_8_bot, y, 9)
    end
end

@inline function signflip(v::Vec{N,T}, mask::Vec{N,UInt32}) where {N, T<:AbstractFloat}
    # Reinterpret float as uint, XOR with sign bit, reinterpret back.
    # Julia doesn't like bitwise operations on its floats...
    v_uint = reinterpret(Vec{N,UInt32}, v)
    v_flipped = v_uint ⊻ mask
    return reinterpret(Vec{N,T}, v_flipped)
end