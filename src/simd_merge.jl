# ============================================================================
# PATTERN-BASED SIMD OPERATION FUSION (Version 3)
# ============================================================================
#
# Philosophy: Analyze the PATTERN of operation types, not specific values.
# Fuse based on what combinations make sense, avoid lookup tables.
#
# Key improvements over V2:
# - Pattern-based dispatch: (n_identity, n_imaginary, n_trig, n_sqrt2)
# - True heterogeneous fusion when beneficial
# - Avoids generating code for patterns that won't occur
# - Selective SIMD operations on lanes
#
# ACTUAL FFT TWIDDLE PATTERNS:
# For FFT of size n=2^q, recursive butterfly stages generate twiddles for
# k ∈ [0, 1, 2, ..., n/2-1] which produce:
# - Always "1" at k=0 (identity)
# - "im", "-im" at specific k (imaginary)
# - "INV_SQRT2_*" variants at n/8, 3n/8, 5n/8, 7n/8 (sqrt2)
# - (num, den, quadrant) tuples for general k (trigonometric)
#
# This means HETEROGENEOUS patterns (mixed types) are COMMON!
#
# FUSION STRATEGY:
# - Write scripts for TYPES of arguments (all imaginary, all trig, etc.)
# - Do NOT write lookup tables for specific single-argument combinations
# - Handle permutations by grouping operations of the same type
# - Support common heterogeneous patterns (identity+imaginary, imaginary+trig)
#
# ============================================================================

"""Minimal operation representation for SIMD fusion"""
struct Op
    output::String
    input::String
    twiddle::Any
end

# Export the main interface
export Op, merge_ops

# ============================================================================
# TYPE CLASSIFICATION - Simple predicates, no LUTs
# ============================================================================

is_identity(twiddle) = twiddle === nothing || twiddle == "1"
is_negate(twiddle) = twiddle == "-1"
is_imaginary(twiddle) = twiddle in ["im", "-im"]
is_trig(twiddle) = twiddle isa Tuple && length(twiddle) == 3
is_sqrt2(twiddle) = twiddle in ["INV_SQRT2_Q4", "-INV_SQRT2_Q1"]

"""Classify twiddle into broad category"""
function classify_twiddle(twiddle)
    is_identity(twiddle) && return :identity
    is_negate(twiddle) && return :negate
    is_imaginary(twiddle) && return :imaginary
    is_trig(twiddle) && return :trig
    is_sqrt2(twiddle) && return :sqrt2
    return :unknown
end

# ============================================================================
# PATTERN ANALYSIS - Count operation types
# ============================================================================

"""
Analyze operations and return pattern signature.
Returns: (n_identity, n_negate, n_imaginary, n_trig, n_sqrt2, n_unknown)
"""
function analyze_pattern(ops::Vector{Op})
    n_identity = count(op -> is_identity(op.twiddle), ops)
    n_negate = count(op -> is_negate(op.twiddle), ops)
    n_imaginary = count(op -> is_imaginary(op.twiddle), ops)
    n_trig = count(op -> is_trig(op.twiddle), ops)
    n_sqrt2 = count(op -> is_sqrt2(op.twiddle), ops)
    n_unknown = length(ops) - (n_identity + n_negate + n_imaginary + n_trig + n_sqrt2)

    return (n_identity, n_negate, n_imaginary, n_trig, n_sqrt2, n_unknown)
end

# ============================================================================
# HIERARCHICAL VECTOR MERGING (from V2)
# ============================================================================

function merge_vectors_tree(input_names::Vector{String}, floats_per_vec::Int; var_prefix="tmp")
    n = length(input_names)
    if n == 1
        return (String[], input_names[1])
    elseif n == 2
        width = 2 * floats_per_vec
        indices = join(0:width-1, ", ")
        var = "$(var_prefix)_w$(width)"
        code = ["$var = shufflevector($(input_names[1]), $(input_names[2]), Val(($indices)))"]
        return (code, var)
    else
        mid = n ÷ 2
        left_code, left_var = merge_vectors_tree(input_names[1:mid], floats_per_vec; var_prefix="$(var_prefix)_L")
        right_code, right_var = merge_vectors_tree(input_names[mid+1:end], floats_per_vec; var_prefix="$(var_prefix)_R")

        code_lines = vcat(left_code, right_code)
        total_width = n * floats_per_vec
        indices = join(0:total_width-1, ", ")
        final_var = "$(var_prefix)_w$(total_width)"
        push!(code_lines, "$final_var = shufflevector($left_var, $right_var, Val(($indices)))")

        return (code_lines, final_var)
    end
end

function gen_signflip_mask(pattern::Vector{Bool}, width::Int, ::Type{T}) where T
    UIntType = T == Float32 ? "UInt32" : (T == Float64 ? "UInt64" : "UInt16")
    sign_bit = T == Float32 ? "0x80000000" : (T == Float64 ? "0x8000000000000000" : "0x8000")
    zero_bit = T == Float32 ? "0x00000000" : (T == Float64 ? "0x0000000000000000" : "0x0000")

    mask_vals = [pattern[(i-1) % length(pattern) + 1] ? sign_bit : zero_bit for i in 1:width]
    return "Vec{$width,$UIntType}(($(join(mask_vals, ", "))))"
end

function gen_extract(merged_var::String, n_ops::Int, floats_per_vec::Int; swap_pattern=nothing)
    parts = String[]
    for i in 1:n_ops
        start_idx = (i-1) * floats_per_vec
        needs_swap = !isnothing(swap_pattern) && swap_pattern[i]

        if needs_swap
            indices = join(reverse(start_idx:(start_idx + floats_per_vec - 1)), ", ")
        else
            indices = join(start_idx:(start_idx + floats_per_vec - 1), ", ")
        end
        push!(parts, "shufflevector($merged_var, Val(($indices)))")
    end
    return parts
end

# ============================================================================
# SINGLE OPERATION HANDLERS - Handle individual operations standalone
# ============================================================================

"""Generate code for a single imaginary operation (im or -im)"""
function gen_single_imaginary(op::Op, ::Type{T}, floats_per_vec::Int) where T
    if op.twiddle == "im"
        # i*(r+ii) = -ir + i = i - ir → (r,i) -> (-i,r)
        # signflip imag, then swap: (r,i) -> (r,-i) -> (-i,r)
        sign_mask = gen_signflip_mask([false, true], floats_per_vec, T)
        return "$(op.output) = shufflevector(signflip($(op.input), $sign_mask), Val((1, 0)))"
    elseif op.twiddle == "-im"
        # -i*(r+ii) = -(-ir + i) = ir - i = i(r-i) → (r,i) -> (i,-r)
        # signflip real, then swap: (r,i) -> (-r,i) -> (i,-r)
        sign_mask = gen_signflip_mask([true, false], floats_per_vec, T)
        return "$(op.output) = shufflevector(signflip($(op.input), $sign_mask), Val((1, 0)))"
    else
        error("Unknown imaginary twiddle: $(op.twiddle)")
    end
end

"""Generate code for a single trigonometric operation"""
function gen_single_trig(op::Op, ::Type{T}, floats_per_vec::Int) where T
    num, den, quadrant = op.twiddle
    c = "COSPI_$(num)_$(den)"
    s = "SINPI_$(num)_$(den)"

    num_complex = floats_per_vec ÷ 2
    r_indices = join([2i for i in 0:num_complex-1], ", ")
    i_indices = join([2i+1 for i in 0:num_complex-1], ", ")

    interleave = join(vcat([[i, i + num_complex] for i in 0:num_complex-1]...), ", ")

    # Handle both enum values and symbols for backward compatibility
    quad_val = quadrant isa Symbol ? quadrant : Symbol(string(quadrant))

    if quad_val == :Q1
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(v_r, $c, -v_i * $s)
            out_i = muladd(v_r, $s, v_i * $c)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :Q4
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(v_r, $c, v_i * $s)
            out_i = muladd(-v_r, $s, v_i * $c)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :ImQ4
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(v_r, $s, -v_i * $c)
            out_i = muladd(v_r, $c, v_i * $s)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :ImQ1
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(-v_r, $s, -v_i * $c)
            out_i = muladd(v_r, $c, -v_i * $s)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :NegQ1
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(-v_r, $c, v_i * $s)
            out_i = muladd(-v_r, $s, -v_i * $c)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :NegQ4
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(-v_r, $c, -v_i * $s)
            out_i = muladd(v_r, $s, -v_i * $c)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :NegImQ1
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(v_r, $s, v_i * $c)
            out_i = muladd(-v_r, $c, v_i * $s)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    elseif quad_val == :NegImQ4
        return """$(op.output) = (let
            v_r = shufflevector($(op.input), Val(($r_indices)))
            v_i = shufflevector($(op.input), Val(($i_indices)))
            out_r = muladd(-v_r, $s, v_i * $c)
            out_i = muladd(-v_r, $c, -v_i * $s)
            shufflevector(out_r, out_i, Val(($interleave)))
        end)"""
    else
        error("Unknown quadrant: $quad_val (original: $quadrant)")
    end
end

"""Generate code for a single sqrt2 operation"""
function gen_single_sqrt2(op::Op, ::Type{T}, floats_per_vec::Int) where T
    if op.twiddle == "INV_SQRT2_Q4"
        return """$(op.output) = (let
            v = $(op.input)
            v_swap = shufflevector(v, Val((1, 0)))
            v_sum = v + v_swap
            v_diff = v - v_swap
            shufflevector(v_sum, v_diff, Val((0, 3))) * INV_SQRT2
        end)"""
    elseif op.twiddle == "-INV_SQRT2_Q1"
        return """$(op.output) = (let
            v_swap = shufflevector($(op.input), Val((1, 0)))
            v_sum = $(op.input) + v_swap
            v_diff = $(op.input) - v_swap
            shufflevector(v_diff, -v_sum, Val((1, 2))) * INV_SQRT2
        end)"""
    else
        error("Unknown sqrt2 twiddle: $(op.twiddle)")
    end
end

# ============================================================================
# HOMOGENEOUS FUSERS - Handle single operation type
# ============================================================================

"""Fuse operations where all are IMAGINARY (im, -im, any mix)"""
function fuse_all_imaginary(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int) where T
    n = length(ops)

    # For single operation, use scalar generation to avoid unnecessary complexity
    if n == 1
        return gen_single_imaginary(ops[1], T, floats_per_vec)
    end

    input_names = [op.input for op in ops]
    output_names = [op.output for op in ops]

    # Build sign pattern based on specific twiddle (im vs -im)
    sign_patterns = []
    swap_required = Bool[]

    for op in ops
        if op.twiddle == "im"
            push!(sign_patterns, Bool[false, true])  # Flip imag: (r,i)->(r,-i)->(swap)->(-i,r)
            push!(swap_required, true)
        elseif op.twiddle == "-im"
            push!(sign_patterns, Bool[true, false])  # Flip real: (r,i)->(-r,i)->(swap)->(i,-r)
            push!(swap_required, true)
        end
    end

    # Combine patterns
    combined_pattern = Bool[]
    for pattern in sign_patterns
        append!(combined_pattern, repeat(pattern, floats_per_vec ÷ length(pattern)))
    end

    code_lines = String[]
    merge_code, merged_var = merge_vectors_tree(input_names, floats_per_vec)
    append!(code_lines, merge_code)

    if any(combined_pattern)
        target_width = n * floats_per_vec
        mask = gen_signflip_mask(combined_pattern, target_width, T)
        push!(code_lines, "flipped = signflip($merged_var, $mask)")
        merged_var = "flipped"
    end

    extracts = gen_extract(merged_var, n, floats_per_vec; swap_pattern=swap_required)

    if n == 1
        # Single operation - wrap in let block to avoid variable scope issues
        return "$(output_names[1]) = (let\n    " *
               join(code_lines, "\n    ") * "\n    $(extracts[1])\nend)"
    else
        return "($(join(output_names, ", "))) = (let\n    " *
               join(code_lines, "\n    ") * "\n    (\n    " *
               join(extracts, ",\n    ") * "\n    )\nend)"
    end
end

"""Fuse operations where all are TRIG (cispi with any values)"""
function fuse_all_trig(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int) where T
    n = length(ops)

    # For single operation, use scalar generation to avoid Vec{1,T} overhead
    if n == 1
        return gen_single_trig(ops[1], T, floats_per_vec)
    end

    input_names = [op.input for op in ops]
    output_names = [op.output for op in ops]
    num_complex_per_op = floats_per_vec ÷ 2
    total_complex = n * num_complex_per_op

    # Bake signs into constants - NO separate sign vectors!
    # out_r = muladd(v_r, cos_r, v_i .* sin_i)
    # out_i = muladd(v_r, sin_r, v_i .* cos_i)
    cos_r_vals = String[]  # cos values for out_r (with sign)
    sin_i_vals = String[]  # sin values for out_i term (with sign)
    sin_r_vals = String[]  # sin values for out_i FMA (with sign)
    cos_i_vals = String[]  # cos values for out_i term (with sign)

    for op in ops
        num, den, quadrant = op.twiddle
        quad_val = quadrant isa Symbol ? quadrant : Symbol(string(quadrant))

        cos_sym = "COSPI_$(num)_$(den)"
        sin_sym = "SINPI_$(num)_$(den)"

        # Map quadrants - bake signs directly into constant names
        # Base formula: (r+ii)*(c+si) = (rc - si) + i(rs + ic)
        if quad_val == :Q1  # cispi(θ) = cos + i*sin
            # out_r = v_r * c - v_i * s  =  muladd(v_r, c, v_i * (-s))
            # out_i = v_r * s + v_i * c  =  muladd(v_r, s, v_i * c)
            r_c, i_s, r_s, i_c = cos_sym, "-$sin_sym", sin_sym, cos_sym
        elseif quad_val == :Q4  # cispi(-θ) = cos - i*sin
            # out_r = v_r * c + v_i * s
            # out_i = -v_r * s + v_i * c
            r_c, i_s, r_s, i_c = 1.0, 1.0, -1.0, 1.0
        elseif quad_val == :ImQ1  # i*cispi(θ) = -sin + i*cos
            # out_r = -v_r * s - v_i * c
            # out_i = v_r * c - v_i * s
            r_c, i_s, r_s, i_c = -1.0, -1.0, 1.0, -1.0
            cos_sym, sin_sym = sin_sym, cos_sym  # Swap roles
        elseif quad_val == :ImQ4  # i*cispi(-θ) = sin + i*cos
            # out_r = v_r * s - v_i * c
            # out_i = v_r * c + v_i * s
            r_c, i_s, r_s, i_c = 1.0, -1.0, 1.0, 1.0
            cos_sym, sin_sym = sin_sym, cos_sym  # Swap roles
        elseif quad_val == :NegQ1  # -cispi(θ) = -cos - i*sin
            # out_r = -v_r * c + v_i * s
            # out_i = -v_r * s - v_i * c
            r_c, i_s, r_s, i_c = -1.0, 1.0, -1.0, -1.0
        elseif quad_val == :NegQ4  # -cispi(-θ) = -cos + i*sin
            # out_r = -v_r * c - v_i * s
            # out_i = v_r * s - v_i * c
            r_c, i_s, r_s, i_c = -1.0, -1.0, 1.0, -1.0
        elseif quad_val == :NegImQ1  # -i*cispi(θ) = sin - i*cos
            # out_r = v_r * s + v_i * c
            # out_i = -v_r * c + v_i * s
            r_c, i_s, r_s, i_c = -1.0, 1.0, 1.0, 1.0
            cos_sym, sin_sym = sin_sym, cos_sym  # Swap roles
        elseif quad_val == :NegImQ4  # -i*cispi(-θ) = -sin - i*cos
            # out_r = -v_r * s + v_i * c
            # out_i = -v_r * c - v_i * s
            r_c, i_s, r_s, i_c = -1.0, 1.0, -1.0, -1.0
            cos_sym, sin_sym = sin_sym, cos_sym  # Swap roles
        else
            error("Unknown quadrant in fuse_all_trig: $quad_val")
        end

        for _ in 1:num_complex_per_op
            push!(cos_syms, cos_sym)
            push!(sin_syms, sin_sym)
            push!(r_cos_signs, r_c)
            push!(i_sin_signs, i_s)
            push!(r_sin_signs, r_s)
            push!(i_cos_signs, i_c)
        end
    end

    code_lines = String[]
    merge_code, merged_var = merge_vectors_tree(input_names, floats_per_vec)
    append!(code_lines, merge_code)

    r_indices = join([2i for i in 0:total_complex-1], ", ")
    i_indices = join([2i+1 for i in 0:total_complex-1], ", ")
    push!(code_lines, "v_r = shufflevector($merged_var, Val(($r_indices)))")
    push!(code_lines, "v_i = shufflevector($merged_var, Val(($i_indices)))")
    push!(code_lines, "cos_vec = Vec{$total_complex,$T}(($(join(cos_syms, ", "))))")
    push!(code_lines, "sin_vec = Vec{$total_complex,$T}(($(join(sin_syms, ", "))))")
    push!(code_lines, "r_cos_sign = Vec{$total_complex,$T}(($(join(r_cos_signs, ", "))))")
    push!(code_lines, "i_sin_sign = Vec{$total_complex,$T}(($(join(i_sin_signs, ", "))))")
    push!(code_lines, "r_sin_sign = Vec{$total_complex,$T}(($(join(r_sin_signs, ", "))))")
    push!(code_lines, "i_cos_sign = Vec{$total_complex,$T}(($(join(i_cos_signs, ", "))))")
    push!(code_lines, "out_r = (v_r .* cos_vec .* r_cos_sign) .+ (v_i .* sin_vec .* i_sin_sign)")
    push!(code_lines, "out_i = (v_r .* sin_vec .* r_sin_sign) .+ (v_i .* cos_vec .* i_cos_sign)")

    interleave = Int[]
    for i in 0:total_complex-1
        push!(interleave, i, i + total_complex)
    end
    push!(code_lines, "result = shufflevector(out_r, out_i, Val(($(join(interleave, ", ")))))")

    extracts = gen_extract("result", n, floats_per_vec)

    if n == 1
        # Single operation - wrap in let block to avoid variable scope issues
        return "$(output_names[1]) = (let\n    " *
               join(code_lines, "\n    ") * "\n    $(extracts[1])\nend)"
    else
        return "($(join(output_names, ", "))) = (let\n    " *
               join(code_lines, "\n    ") * "\n    (\n    " *
               join(extracts, ",\n    ") * "\n    )\nend)"
    end
end

"""Fuse operations where all are SQRT2 (any mix of variants)"""
function fuse_all_sqrt2(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int) where T
    n = length(ops)

    # For single operation, use scalar generation to avoid unnecessary complexity
    if n == 1
        return gen_single_sqrt2(ops[1], T, floats_per_vec)
    end

    input_names = [op.input for op in ops]
    output_names = [op.output for op in ops]

    code_lines = String[]
    merge_code, merged_var = merge_vectors_tree(input_names, floats_per_vec)
    append!(code_lines, merge_code)

    target_width = n * floats_per_vec
    swap_indices = Int[]
    for i in 0:floats_per_vec:target_width-1
        push!(swap_indices, i+1, i)
    end

    push!(code_lines, "v_swap = shufflevector($merged_var, Val(($(join(swap_indices, ", ")))))")
    push!(code_lines, "v_sum = $merged_var + v_swap")
    push!(code_lines, "v_diff = $merged_var - v_swap")

    result_indices = Int[]
    neg_pattern = Bool[]

    for (i, op) in enumerate(ops)
        base = (i-1) * floats_per_vec
        if op.twiddle == "INV_SQRT2_Q4"
            push!(result_indices, base, base + 1 + target_width)
            push!(neg_pattern, false, false)
        else  # -INV_SQRT2_Q1
            push!(result_indices, base + target_width, base + 1)
            push!(neg_pattern, false, true)
        end
    end

    if any(neg_pattern)
        combined_width = 2 * target_width
        mask = gen_signflip_mask(neg_pattern, combined_width, T)
        push!(code_lines, "v_combined = shufflevector(v_sum, v_diff, Val(($(join(0:combined_width-1, ", ")))))")
        push!(code_lines, "v_adjusted = signflip(v_combined, $mask)")
        push!(code_lines, "result = shufflevector(v_adjusted, Val(($(join(result_indices, ", "))))) * INV_SQRT2")
    else
        push!(code_lines, "result = shufflevector(v_sum, v_diff, Val(($(join(result_indices, ", "))))) * INV_SQRT2")
    end

    extracts = gen_extract("result", n, floats_per_vec)

    if n == 1
        # Single operation - wrap in let block to avoid variable scope issues
        return "$(output_names[1]) = (let\n    " *
               join(code_lines, "\n    ") * "\n    $(extracts[1])\nend)"
    else
        return "($(join(output_names, ", "))) = (let\n    " *
               join(code_lines, "\n    ") * "\n    (\n    " *
               join(extracts, ",\n    ") * "\n    )\nend)"
    end
end

# ============================================================================
# HETEROGENEOUS FUSERS - Handle mixed operation types
# ============================================================================

"""
Fuse identity/negate + imaginary operations.
Identity/negate ops pass through (with sign), imaginary ops get signflip + swap.
"""
function fuse_identity_imaginary(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int) where T
    # Separate into identity/negate and imaginary groups
    identity_ops = [op for op in ops if is_identity(op.twiddle) || is_negate(op.twiddle)]
    imaginary_ops = [op for op in ops if is_imaginary(op.twiddle)]

    code_parts = String[]

    # Identity/negate operations - simple pass-through with optional negation
    if !isempty(identity_ops)
        for op in identity_ops
            prefix = is_negate(op.twiddle) ? "-" : ""
            push!(code_parts, "$(op.output) = $(prefix)$(op.input)")
        end
    end

    # Imaginary operations - fuse together
    if !isempty(imaginary_ops)
        push!(code_parts, fuse_all_imaginary(imaginary_ops, T, floats_per_vec))
    end

    return join(code_parts, "\n") * "\n"
end

"""
Fuse imaginary + trig operations.
Both groups get their own merged operations.
"""
function fuse_imaginary_trig(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int) where T
    imaginary_ops = [op for op in ops if is_imaginary(op.twiddle)]
    trig_ops = [op for op in ops if is_trig(op.twiddle)]

    code_parts = String[]

    if !isempty(imaginary_ops)
        push!(code_parts, fuse_all_imaginary(imaginary_ops, T, floats_per_vec))
    end

    if !isempty(trig_ops)
        push!(code_parts, fuse_all_trig(trig_ops, T, floats_per_vec))
    end

    return join(code_parts, "\n\n") * "\n"
end

# ============================================================================
# PATTERN-BASED DISPATCH
# ============================================================================

function can_merge(n_ops::Int, floats_per_vec::Int, ::Type{T}, simd_bits::Int) where T
    total_bits = n_ops * floats_per_vec * sizeof(T) * 8
    return total_bits <= simd_bits
end

"""
Main entry point: Pattern-based fusion.

Analyzes the pattern of operation types and dispatches to appropriate fuser.
"""
function merge_ops(ops::Vector{Op}, ::Type{T}, floats_per_vec::Int, simd_bits::Int=256) where T
    if isempty(ops)
        return ""
    end

    n = length(ops)

    # Check SIMD width
    if !can_merge(n, floats_per_vec, T, simd_bits)
        # Can't fit - generate individually
        code_parts = String[]
        for op in ops
            single = merge_ops([op], T, floats_per_vec, simd_bits)
            push!(code_parts, single)
        end
        return join(code_parts, "\n") * "\n"
    end

    # Analyze pattern
    (n_identity, n_negate, n_imaginary, n_trig, n_sqrt2, n_unknown) = analyze_pattern(ops)

    # Dispatch based on pattern
    # HOMOGENEOUS patterns
    if n_identity == n
        # All identity - simple pass-through
        return join(["$(op.output) = $(op.input)" for op in ops], "\n")

    elseif n_negate == n
        # All negate - simple negation
        return join(["$(op.output) = -$(op.input)" for op in ops], "\n")

    elseif n_imaginary == n
        # All imaginary - single fused signflip
        return fuse_all_imaginary(ops, T, floats_per_vec)

    elseif n_trig == n
        # All trig - single fused complex multiply
        return fuse_all_trig(ops, T, floats_per_vec)

    elseif n_sqrt2 == n
        # All sqrt2 - single fused sqrt2 operation
        return fuse_all_sqrt2(ops, T, floats_per_vec)

    # HETEROGENEOUS patterns - common cases only
    elseif (n_identity > 0 || n_negate > 0) && n_imaginary > 0 && n_trig == 0 && n_sqrt2 == 0
        # Identity/negate + imaginary (common in first stages)
        return fuse_identity_imaginary(ops, T, floats_per_vec)

    elseif n_identity == 0 && n_negate == 0 && n_imaginary > 0 && n_trig > 0 && n_sqrt2 == 0
        # Imaginary + trig (common in middle stages)
        return fuse_imaginary_trig(ops, T, floats_per_vec)

    elseif (n_identity > 0 || n_negate > 0) && n_imaginary == 0 && n_trig > 0 && n_sqrt2 == 0
        # Identity/negate + trig
        code_parts = String[]
        for op in ops
            if is_identity(op.twiddle) || is_negate(op.twiddle)
                prefix = is_negate(op.twiddle) ? "-" : ""
                push!(code_parts, "$(op.output) = $(prefix)$(op.input)")
            end
        end
        trig_ops = [op for op in ops if is_trig(op.twiddle)]
        if !isempty(trig_ops)
            push!(code_parts, fuse_all_trig(trig_ops, T, floats_per_vec))
        end
        return join(code_parts, "\n") * "\n"

    else
        # Complex heterogeneous - IMPROVED STRATEGY
        # 1. Group operations by type
        # 2. Fuse each group (even if small - KEY FIX for Test 7)
        # 3. Recombine all results into wide_result

        identity_ops = [op for op in ops if is_identity(op.twiddle) || is_negate(op.twiddle)]
        imaginary_ops = [op for op in ops if is_imaginary(op.twiddle)]
        trig_ops = [op for op in ops if is_trig(op.twiddle)]
        sqrt2_ops = [op for op in ops if is_sqrt2(op.twiddle)]

        code_parts = String[]

        # Handle identity/negate (passthrough)
        for op in identity_ops
            prefix = is_negate(op.twiddle) ? "-" : ""
            push!(code_parts, "$(op.output) = $(prefix)$(op.input)")
        end

        # Fuse imaginary operations
        if !isempty(imaginary_ops)
            if length(imaginary_ops) > 1
                push!(code_parts, fuse_all_imaginary(imaginary_ops, T, floats_per_vec))
            else
                push!(code_parts, gen_single_imaginary(imaginary_ops[1], T, floats_per_vec))
            end
        end

        # **KEY FIX**: Fuse trig operations even if there are only 2
        if !isempty(trig_ops)
            if length(trig_ops) >= 2
                # Use vectorized fusion for 2+ operations
                push!(code_parts, fuse_all_trig(trig_ops, T, floats_per_vec))
            else
                push!(code_parts, gen_single_trig(trig_ops[1], T, floats_per_vec))
            end
        end

        # Fuse sqrt2 operations
        if !isempty(sqrt2_ops)
            if length(sqrt2_ops) > 1
                push!(code_parts, fuse_all_sqrt2(sqrt2_ops, T, floats_per_vec))
            else
                push!(code_parts, gen_single_sqrt2(sqrt2_ops[1], T, floats_per_vec))
            end
        end

        return join(code_parts, "\n") * "\n"
    end
end

