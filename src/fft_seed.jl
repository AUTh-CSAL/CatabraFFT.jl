include("suffix.jl")
include("radix_plan.jl")
using SIMD

const SIMD_BITS = 256

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
    vec_width = 2sizeof(T)  # Width in bytes for real+imag pair
    
    # mode = :default
    # Use direct indexing into reinterpreted Vector{T} array
    join([
        let
            m = match(r"(\d+)\D*$", s)
            num = parse(Int, m.captures[1])
            var = startswith(s, "x") ? "x" :
                  startswith(s, "y") ? "y" :
                  startswith(s, "D") ? "d" : error("Unknown input: $s")
            # Direct indexing: px[2*i-1] for real, px[2*i] for imag
            prefix = i == 1 ? "" : " "
            "$(prefix)$(var)$(num)_r , $(prefix)$(var)$(num)_i = $(input)[$(2*num-1)], $(input)[$(2*num)]"
        end
        for (i, s) in enumerate(t)
    ], "; ")
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

# Overload for when D is a type (no actual D matrix)
function makefftradix(n::Int, suffixes::SuffixFlags, ::Type{Vector{T}}, p::Int, op, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat
    # Call with empty vector when no D matrix is needed
    return makefftradix(n, suffixes, Union{String, Twiddle}[], p, op, SIZE, T, SIMD_BITS)
end

# Wrapper for any other kernel shell strategy planer
function makefftradix(n::Int, suffixes::SuffixFlags, D::Union{Vector{Union{String, Twiddle}}, Vector{T}}, p::Int, op, SIZE::Int, ::Type{T}, SIMD_BITS) where T <: AbstractFloat
    global inc = inccounter()
    
    mode = :default # ALL ME

    input, output = String(op.input_buffer), String(op.output_buffer)
    
    # Key parameters for Stockham algorithm
    radix = n 
    stride = op.stride
    input_spacing = SIZE ÷ radix  # Spacing between input elements in each butterfly
    
    x = mode == :default ? ["$(input)$(p + 1 + (i-1)*input_spacing)" for i in 1:radix] : ["v_all$(p + 1 + (i-1)*input_spacing)" for i in 1:radix] 
    base = 2 * ((p ÷ stride) * (stride * radix) + (p % stride))

    y = ["$output[$(base + 1 + i)]" for i in 0:2*radix-1]
    @show x, y
    
    d = isempty(D) ? nothing : D
    
    px = mode == :vgather ? "p$(input) = reinterpret($T, $(input));" : "" 
    py = ""

    complex_size_bits = 2 * sizeof(T) * 8
    complexes_per_vec = SIMD_BITS ÷ complex_size_bits

    # MOST IMPORTANT LINE!!!
    kernel_code = mode == :default ? recfft2(y, x, d, nothing, true, T, 1, mode, py, input) : recfft2_simd(y, x, d, nothing, true, T, 1, mode, py, complexes_per_vec) 
    kernel_code = "$px\n$kernel_code"
    
    if isempty(kernel_code)
        return quote end
    else
        try
            @show parsed_expr = Meta.parse("begin\n$kernel_code\nend")
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

# Generate twiddle factor expressions for FFT
# w_k = e^(-2πik/n) for k in ks array
function get_twiddle_expression(ks, n; T=Float64, accuracy=nothing)
    twiddles = String[]

    for k in ks
        # Normalize k to 0 <= k < n
        k_norm = mod(k, n)

        # Special cases for common twiddle values
        if k_norm == 0
            push!(twiddles, "1")
        elseif k_norm == n ÷ 4 && n % 4 == 0
            # e^(-iπ/2) = -i
            push!(twiddles, "-im")
        elseif k_norm == n ÷ 8 && n % 8 == 0
            # e^(-iπ/4) = (1-i)/√2
            push!(twiddles, "INV_SQRT2_Q4")
        elseif k_norm == 3n ÷ 8 && n % 8 == 0
            # e^(-3iπ/4) = -(1+i)/√2
            push!(twiddles, "-INV_SQRT2_Q1")
        else
            # General case: CISPI format
            # w_k = e^(-2πik/n) = cos(-2πk/n) + i*sin(-2πk/n)
            # Simplify the fraction k/n
            g = gcd(k_norm, n)
            num = k_norm ÷ g
            den = n ÷ g

            # Determine quadrant based on angle -2πk/n
            # Q1: 0 to π/2 (but negative angle, so 3π/2 to 2π)
            # Q4: -π/2 to 0 (or 3π/2 to 2π)
            angle_frac = k_norm / n  # fraction of full rotation

            if angle_frac <= 0.25
                # Quadrant 4: small negative angles
                q = "Q4"
            elseif angle_frac <= 0.5
                # Quadrant 1 or 4 depending on sign convention
                q = "Q1"
            elseif angle_frac <= 0.75
                q = "Q1"
            else
                q = "Q4"
            end

            push!(twiddles, "CISPI_$(num)_$(den)_$(q)")
        end
    end

    return twiddles
end
#
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
    floats_per_vec = 2 * complexes_per_vec

    if n == 1
        return ""

    elseif n == 2
        # BASE CASE: Mirror recfft2's n=2 logic with Vec operations
        s = if !isnothing(d)
            if isnothing(w)
                if root
                    load_code = load_gen_simd(x; mode=mode, T=T, ptr_name="px", SIMD_BITS=SIMD_BITS)
                    store_code = store_gen_simd(y, ["v_sum", "v_diff"]; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)
                    """
                    $load_code
                    tmp0 = $(x[1]) - $(x[2])
                    $py
                    v_sum = $(x[1]) + $(x[2])
                    v_diff = $(vec_mul_twiddle_d("tmp0", 1, 2, T, floats_per_vec))
                    $store_code
                    """
                end
            end
        else
            if root
                load_code = load_gen_simd(x; mode=mode, T=T, ptr_name="px", SIMD_BITS=SIMD_BITS)
                store_code = store_gen_simd(y, ["v_sum", "v_diff"]; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)
                """
                $load_code
                v_sum = $(x[1]) + $(x[2])
                v_diff = $(x[1]) - $(x[2])
                $store_code
                """
            else
                if isnothing(w)
                    """
                    $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(x[1]) - $(x[2])
                    """
                else
                    w[1] == "1" ?
                        """
                        $(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(vec_mul_twiddle("-", x[1], x[2], w[2], T, floats_per_vec))
                        """ :
                        """
                        $(y[1]), $(y[2]) = $(vec_mul_twiddle("+", x[1], x[2], w[1], T, floats_per_vec)), $(vec_mul_twiddle("-", x[1], x[2], w[2], T, floats_per_vec))
                        """
                end
            end
        end

        return something(s, "")

    else
        # RECURSIVE CASE: Mirror recfft2's recursive structure EXACTLY
        n2 = n ÷ 2
        t = ["tvec$i" for i in tmp_base:(tmp_base + n - 1)]
        new_tmp_base = tmp_base + n

        # Recursively handle sub-transforms (mirror recfft2 lines 708-709)
        s1 = recfft2_simd(t[1:n2], x[1:2:n], nothing, nothing, false, T, new_tmp_base, mode, py, complexes_per_vec)
        s2 = recfft2_simd(t[n2+1:n], x[2:2:n], nothing, get_twiddle_expression(collect(0:n2-1), n; T=T, accuracy=nothing), false, T, new_tmp_base, mode, py, complexes_per_vec)

        # Generate tmp declarations for butterfly (mirror recfft2 lines 711-727)
        # Create consecutive tmp variables: tmp0, tmp1, ... for sums, then tmp(n2-1), tmp(n2), ... for diffs
        tmp_decls = if n > 2
            parts = String[]
            idx = 0
            # First create sum temps (corresponding to x1_exprs in scalar)
            for i in 2:n2
                push!(parts, "tmp$(idx) = $(t[i]) + $(t[i+n2])")
                idx += 1
            end
            # Then create diff temps (corresponding to x2_exprs in scalar)
            for i in 2:n2
                push!(parts, "tmp$(idx) = $(t[i]) - $(t[i+n2])")
                idx += 1
            end
            join(parts, "\n") * "\n"
        else
            ""
        end

        # Final layer combining with D matrix or twiddles (mirror recfft2 lines 729-787)
        s3p, s3m = "", ""

        if !isnothing(d)
            if isnothing(w)
                if root
                    # Build vector variables for first half
                    vec_vars_p = ["v_out_$(i-1)" for i in 1:n2]
                    # d[i-1] is twiddle (i-1) -> cos at 2*(i-1)-1 = 2i-3, sin at 2*(i-1) = 2i-2
                    vec_assigns_p = ["$(vec_vars_p[1]) = $(t[1]) + $(t[1+n2])"] *
                                   ["$(vec_vars_p[i]) = $(vec_mul_twiddle_d("tmp$(i-2)", 2*i-3, 2*i-2, T, floats_per_vec))" for i in 2:n2]

                    # Build vector variables for second half
                    vec_vars_m = ["v_out_$(i+n2-1)" for i in 1:n2]
                    # d[n2] -> (2*n2-1, 2*n2), d[i+n2-1] -> (2i+2n2-3, 2i+2n2-2)
                    vec_assigns_m = ["$(vec_vars_m[1]) = $(vec_mul_twiddle_d("-", t[1], "$(t[1+n2])", 2*n2-1, 2*n2, T, floats_per_vec))"] *
                                   ["$(vec_vars_m[i]) = $(vec_mul_twiddle_d("tmp$(i-3+n2)", 2*i+2*n2-3, 2*i+2*n2-2, T, floats_per_vec))" for i in 2:n2]

                    store_code_p = store_gen_simd(y[1:n2], vec_vars_p; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)
                    store_code_m = store_gen_simd(y[n2+1:end], vec_vars_m; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)

                    s3p = "$py\n$tmp_decls" * join(vec_assigns_p, "\n") * "\n$store_code_p\n"
                    s3m = join(vec_assigns_m, "\n") * "\n$store_code_m\n"
                end
            end
        else
            if isnothing(w)
                if root
                    # Build vector variables for first half
                    vec_vars_p = ["v_out_$(i-1)" for i in 1:n2]
                    vec_assigns_p = ["$(vec_vars_p[1]) = $(t[1]) + $(t[1+n2])"] *
                                   ["$(vec_vars_p[i]) = $(t[i]) + $(t[i+n2])" for i in 2:n2]

                    # Build vector variables for second half
                    vec_vars_m = ["v_out_$(i+n2-1)" for i in 1:n2]
                    vec_assigns_m = ["$(vec_vars_m[1]) = $(t[1]) - $(t[1+n2])"] *
                                   ["$(vec_vars_m[i]) = $(t[i]) - $(t[i+n2])" for i in 2:n2]

                    store_code_p = store_gen_simd(y[1:n2], vec_vars_p; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)
                    store_code_m = store_gen_simd(y[n2+1:end], vec_vars_m; mode=:contiguous, T=T, ptr_name="py", SIMD_BITS=SIMD_BITS)

                    s3p = "$py\n" * join(vec_assigns_p, "\n") * "\n$store_code_p\n"
                    s3m = join(vec_assigns_m, "\n") * "\n$store_code_m\n"
                else
                    s3p = "$(y[1])" * join([", $(y[i])" for i in 2:n2]) *
                          " = " *
                          "$(t[1]) + $(t[1+n2])" * join([", $(t[i]) + $(t[i+n2])" for i in 2:n2]) * "\n"
                    s3m = "$(y[n2+1])" * join([", $(y[i+n2])" for i in 2:n2]) *
                          " = " *
                          "$(t[1]) - $(t[1+n2])" * join([", $(t[i]) - $(t[i+n2])" for i in 2:n2]) * "\n"
                end
            else
                # With twiddles (mirror recfft2 lines 759-787)
                s3p = "$tmp_decls" *
                      "$(y[1])" * join([", $(y[i])" for i in 2:n2]) *
                      " = " *
                      (w[1] == "1" ? "$(t[1]) + $(t[1+n2])" : "$(vec_mul_twiddle("tmp_$(t[1])", w[1], T, floats_per_vec))") *
                      join([", $(vec_mul_twiddle("tmp$(i-2)", w[i], T, floats_per_vec))" for i in 2:n2]) * "\n"
                s3m = "$(y[n2+1])" * join([", $(y[i+n2])" for i in 2:n2]) *
                      " = " *
                      "$(vec_mul_twiddle("-", t[1], "$(t[1+n2])", w[n2+1], T, floats_per_vec))" *
                      join([", $(vec_mul_twiddle("tmp$(i-3+n2)", w[n2+i], T, floats_per_vec))" for i in 2:n2]) * "\n"
            end
        end

        return s1 * s2 * s3p * s3m
    end
end

# SIMD code generation helpers for complex twiddle multiplication
# These generate actual SIMD.jl expressions that will be inlined

# Multiply Vec by -i: (a+bi)*(-i) = b - ai
# Implementation: swap real/imag pairs and negate the (now) real parts
function vec_mul_neg_i_code(vec_name::String, floats_per_vec::Int)
    # Generate shuffle indices to swap r,i pairs: [r1,i1,r2,i2,...] -> [i1,r1,i2,r2,...]
    # For 1-based indexing: [1,2,3,4,...] we want [2,1,4,3,...]
    # After subtracting 1 for 0-based: [1,0,3,2,...]
    swap_indices = Int[]
    for i in 1:2:floats_per_vec
        push!(swap_indices, i+1)  # imag component (1-based)
        push!(swap_indices, i)    # real component (1-based)
    end
    # Generate sign mask to negate real parts (now at even indices after swap)
    sign_mask = join(["0x$(i % 2 == 1 ? "00000000" : "80000000")" for i in 1:floats_per_vec], ", ")

    return "shufflevector(signflip($vec_name, Vec{$floats_per_vec,UInt32}(($sign_mask))), Val(($(join(swap_indices .- 1, ", ")))))"
end

# Multiply Vec by (1-i)/√2 (INV_SQRT2_Q4)
function vec_mul_inv_sqrt2_q4_code(vec_name::String, ::Type{T}, floats_per_vec::Int) where T
    # (r+ii) * (1-i)/√2 = (r+i)/√2 + i(i-r)/√2 = ((r+i), (i-r))/√2
    num_complex = floats_per_vec ÷ 2
    # Extract real and imag parts
    r_indices = [2i-1 for i in 1:num_complex]
    i_indices = [2i for i in 1:num_complex]

    # Interleave for final result: r1, i1, r2, i2, ...
    interleave_indices = Int[]
    for i in 0:num_complex-1
        push!(interleave_indices, i)            # from v_sum
        push!(interleave_indices, i + num_complex)  # from v_diff
    end

    """(let
        v = $vec_name
        v_r = shufflevector(v, Val(($(join(r_indices .- 1, ", ")))))
        v_i = shufflevector(v, Val(($(join(i_indices .- 1, ", ")))))
        v_sum = v_r + v_i
        v_diff = v_i - v_r
        shufflevector(v_sum, v_diff, Val(($(join(interleave_indices, ", "))))) * Vec{$floats_per_vec,$T}(($(join(["INV_SQRT2" for _ in 1:floats_per_vec], ", "))))
    end)"""
end

# Multiply Vec by -(1+i)/√2 (NEG_INV_SQRT2_Q1)
function vec_mul_neg_inv_sqrt2_q1_code(vec_name::String, ::Type{T}, floats_per_vec::Int) where T
    num_complex = floats_per_vec ÷ 2
    r_indices = [2i-1 for i in 1:num_complex]
    i_indices = [2i for i in 1:num_complex]

    # Interleave for final result: r1, i1, r2, i2, ...
    interleave_indices = Int[]
    for i in 0:num_complex-1
        push!(interleave_indices, i)            # from v_diff
        push!(interleave_indices, i + num_complex)  # from -v_sum
    end

    """(let
        v = $vec_name
        v_r = shufflevector(v, Val(($(join(r_indices .- 1, ", ")))))
        v_i = shufflevector(v, Val(($(join(i_indices .- 1, ", ")))))
        v_sum = v_r + v_i
        v_diff = v_i - v_r
        shufflevector(v_diff, -v_sum, Val(($(join(interleave_indices, ", "))))) * Vec{$floats_per_vec,$T}(($(join(["INV_SQRT2" for _ in 1:floats_per_vec], ", "))))
    end)"""
end

# Multiply Vec by general CISPI twiddle: cos(π*n/d) ± i*sin(π*n/d)
function vec_mul_cispi_code(vec_name::String, cos_const::String, sin_const::String, q1::Bool, ::Type{T}, floats_per_vec::Int) where T
    num_complex = floats_per_vec ÷ 2
    r_indices = [2i-1 for i in 1:num_complex]
    i_indices = [2i for i in 1:num_complex]

    # Interleave for final result: r1, i1, r2, i2, ...
    interleave_indices = Int[]
    for i in 0:num_complex-1
        push!(interleave_indices, i)            # from out_r
        push!(interleave_indices, i + num_complex)  # from out_i
    end

    # Complex multiply: (r+ii)*(c+si) = (rc-si) + i(ri+sc)  if q1
    #                   (r+ii)*(c-si) = (rc+si) + i(ri-sc)  if !q1
    if q1
        """(let
            v = $vec_name
            v_r = shufflevector(v, Val(($(join(r_indices .- 1, ", ")))))
            v_i = shufflevector(v, Val(($(join(i_indices .- 1, ", ")))))
            c = $cos_const
            s = $sin_const
            out_r = v_r * c - v_i * s
            out_i = v_r * s + v_i * c
            shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
        end)"""
    else
        """(let
            v = $vec_name
            v_r = shufflevector(v, Val(($(join(r_indices .- 1, ", ")))))
            v_i = shufflevector(v, Val(($(join(i_indices .- 1, ", ")))))
            c = $cos_const
            s = $sin_const
            out_r = v_r * c + v_i * s
            out_i = v_r * (-s) + v_i * c
            shufflevector(out_r, out_i, Val(($(join(interleave_indices, ", ")))))
        end)"""
    end
end

# Update vec_mul_twiddle to call code generators
# Version using numeric d vector: apply twiddle from d[c_idx + i*d[s_idx]] to vector
function vec_mul_twiddle_d(vec_name::String, c_idx::Int, s_idx::Int, ::Type{T}, floats_per_vec::Int) where T
    # Use d[c_idx] for cos, d[s_idx] for sin in vectorized complex multiplication
    c = "d[$c_idx]"
    s = "d[$s_idx]"
    # Call the same CISPI multiplication code, but with direct references to d
    return vec_mul_cispi_code(vec_name, c, s, false, T, floats_per_vec)  # false = Q4 quadrant (cos - i*sin)
end

# Butterfly version: apply twiddle from d vector to (vec1 op vec2)
function vec_mul_twiddle_d(op::String, vec1::String, vec2::String, c_idx::Int, s_idx::Int, ::Type{T}, floats_per_vec::Int) where T
    # Apply twiddle to the result of the butterfly operation
    return vec_mul_twiddle_d("($vec1 $op $vec2)", c_idx, s_idx, T, floats_per_vec)
end

function vec_mul_twiddle(vec_name::String, twiddle::String, ::Type{T}, floats_per_vec::Int) where T
    if twiddle == "1"
        return vec_name
    elseif twiddle == "-im"
        return vec_mul_neg_i_code(vec_name, floats_per_vec)
    elseif twiddle == "INV_SQRT2_Q4"
        return vec_mul_inv_sqrt2_q4_code(vec_name, T, floats_per_vec)
    elseif twiddle == "-INV_SQRT2_Q1"
        return vec_mul_neg_inv_sqrt2_q1_code(vec_name, T, floats_per_vec)
    else
        # General CISPI twiddle
        parsed = parse_cispi(twiddle)
        c = "COSPI_$(parsed.num)_$(parsed.den)"
        s = "SINPI_$(parsed.num)_$(parsed.den)"
        return vec_mul_cispi_code(vec_name, c, s, parsed.q1, T, floats_per_vec)
    end
end

# Butterfly version: apply twiddle to (vec1 op vec2)
function vec_mul_twiddle(op::String, vec1::String, vec2::String, twiddle::String, ::Type{T}, floats_per_vec::Int) where T
    if twiddle == "1"
        return "$vec1 $op $vec2"
    else
        # Apply twiddle to the result of the butterfly operation
        return vec_mul_twiddle("($vec1 $op $vec2)", twiddle, T, floats_per_vec)
    end
end

# Vectorized XOR operation to flip sign at chosen vector elements
@inline function signflip(v::Vec{N,T}, mask::Vec{N,UInt32}) where {N, T<:AbstractFloat}
    # Reinterpret float as uint, XOR with sign bit, reinterpret back
    v_uint = reinterpret(Vec{N,UInt32}, v)
    v_flipped = v_uint ⊻ mask
    return reinterpret(Vec{N,T}, v_flipped)
end

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
