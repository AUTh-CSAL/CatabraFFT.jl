# ============================================================================
# SIMD FFT KERNEL GENERATOR WITH AUTOMATIC TWIDDLE FUSION
# ============================================================================
#
# This is a complete rewrite of recfft2_simd that integrates the automatic
# twiddle fusion mechanism from simd_merge_v3.jl.
#
# Key improvements:
# - Automatic detection and fusion of twiddle operations
# - Wide vector recombination for maximum register efficiency
# - Nested let blocks for compile-time optimization
# - Clean separation of concerns (fusion logic is external)
#
# ============================================================================

"""
recfft2_simd - SIMD FFT kernel generator with automatic twiddle fusion

Generates horizontal SIMD operations with intelligent fusion of twiddle multiplications.
Uses the merge_ops mechanism from simd_merge_v3.jl for optimal vectorization.

Arguments:
- y, x: Output/input variable arrays
- d, w: Twiddle factors (d for DIT, w for general)
- root: Whether this is the root call
- T: Float type (Float32, Float64)
- floats_per_vec: SIMD width in floats (8 for AVX2 Vec{8,Float32})
- tmp_base: Base index for temporary variables
- mode: Load/store mode
- py: Additional store setup code
- complexes_per_vec: Complex numbers per vector
- input_buffer, output_buffer: Buffer names
"""
@inline function recfft2_simd(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base=1,
                              mode=:vgather, py="", complexes_per_vec=4,
                              input_buffer="x", output_buffer="y") where T <: AbstractFloat
    n = length(x)  # Number of complex numbers
    SIMD_WIDTH = floats_per_vec * sizeof(T) * 8

    # BASE CASE: n == 1
    if n == 1
        return ""

    # BASE CASE: n == 2 (Radix-2 butterfly)
    elseif n == 2
        return generate_butterfly2_simd(y, x, d, w, root, T, floats_per_vec,
                                       input_buffer, output_buffer)

    # RECURSIVE CASE: n > 2
    else
        println("n = $n")
        return generate_recursive_fft(y, x, d, w, root, T, floats_per_vec, tmp_base,
                                     mode, py, complexes_per_vec, input_buffer, output_buffer,
                                     SIMD_WIDTH, n)
    end
end

# ============================================================================
# HELPER: Generate n=2 butterfly with twiddles
# ============================================================================

function generate_butterfly2_simd(y, x, d, w, root, ::Type{T}, floats_per_vec,
                                 input_buffer, output_buffer) where T
    # Radix-2 butterfly: y1 = x1 + x2, y2 = x1 - x2 (with twiddles)
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
                $(x[1]) = vload(Vec{2,$T}, $input_buffer, $pos1)
                $(x[2]) = vload(Vec{2,$T}, $input_buffer, $pos2)

                # Butterfly
                tmp0 = $(x[1]) - $(x[2])
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

            # Extract store positions from y array
            store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos2 = parse(Int, match(r"\[(\d+)\]", y[3]).captures[1])

            """
            $(x[1]) = vload(Vec{2,$T}, $input_buffer, $pos1)
            $(x[2]) = vload(Vec{2,$T}, $input_buffer, $pos2)
            t1 = $(x[1]) + $(x[2])
            t2 = $(x[1]) - $(x[2])
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
                # Non-root with twiddles - USE FUSION
                ops = [
                    Op("$(y[1])", "$(x[1]) + $(x[2])", w[1]),
                    Op("$(y[2])", "$(x[1]) - $(x[2])", w[2])
                ]
                return merge_ops(ops, T, 2, 256)
            end
        end
    end
    return something(s, "")
end

# ============================================================================
# HELPER: Generate loads
# ============================================================================

function generate_loads(x, n, mode, ::Type{T}, input_buffer, SIMD_WIDTH, complexes_per_vec) where T
    # Extract indices from variable names (x1 -> 1, x5 -> 5, etc.)
    indices = [parse(Int, match(r"(\d+)", var).captures[1]) for var in x]

    # Check if contiguous
    is_contiguous = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))

    n_floats = 2 * n

    if is_contiguous && n <= complexes_per_vec
        # Case 1: Contiguous access - single vload
        start_idx = 2 * (indices[1] - 1) + 1
        code = "l_all = vload(Vec{$(n_floats),$T}, $input_buffer, $start_idx)\n"

        # Extract each complex number
        for (i, var) in enumerate(x)
            idx_start = 2 * (i - 1)
            shuffle_indices = join([idx_start, idx_start + 1], ", ")
            code *= "$var = shufflevector(l_all, Val(($shuffle_indices)))\n"
        end
        return code
    else
        # Non-contiguous or large: use vgather or multiple vloads
        # For simplicity, generate individual loads for now
        code = ""
        for (i, var) in enumerate(x)
            pos = 2 * (indices[i] - 1) + 1
            code *= "$var = vload(Vec{2,$T}, $input_buffer, $pos)\n"
        end
        return code
    end
end

# ============================================================================
# HELPER: Generate recursive FFT case with twiddle fusion
# ============================================================================

function generate_recursive_fft(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base,
                               mode, py, complexes_per_vec, input_buffer, output_buffer,
                               SIMD_WIDTH, n) where T <: AbstractFloat
    n2 = n ÷ 2
    t = ["tvec$(tmp_base + i - 1)" for i in 1:n]
    new_tmp_base = tmp_base + n

    # 1. Generate loads (root call only)
    load_code = root ? generate_loads(x, n, mode, T, input_buffer, SIMD_WIDTH, complexes_per_vec) : ""

    # 2. Recursive sub-transforms
    s1 = recfft2_simd(t[1:n2], x[1:2:n], nothing, nothing, false, T,
                      floats_per_vec ÷ 2, new_tmp_base, mode, py, complexes_per_vec,
                      input_buffer, output_buffer)

    twiddles_second_half = get_twiddle_expression(collect(0:n2-1), n; T=T, accuracy=nothing)
    s2 = recfft2_simd(t[n2+1:n], x[2:2:n], nothing, twiddles_second_half, false, T,
                      floats_per_vec ÷ 2, new_tmp_base, mode, py, complexes_per_vec,
                      input_buffer, output_buffer)

    # 3. Generate butterfly combinations with TWIDDLE FUSION
    if !isnothing(d)
        # DIT case with D-matrix
        butterfly_code = generate_dit_butterfly_fused(t, d, y, n2, root, T, floats_per_vec, output_buffer, py)
    else
        if isnothing(w)
            # No twiddles - simple butterfly
            butterfly_code = generate_simple_butterfly(t, y, n2, root, T, floats_per_vec, output_buffer, py)
        else
            # General twiddles - **USE FUSION HERE**
            butterfly_code = generate_general_butterfly_fused(t, w, y, n2, root, T, floats_per_vec, output_buffer, py)
        end
    end

    return load_code * s1 * s2 * butterfly_code
end

# ============================================================================
# HELPER: Generate simple butterfly (no twiddles)
# ============================================================================

function generate_simple_butterfly(t, y, n2, root, ::Type{T}, floats_per_vec, output_buffer, py) where T
    if root
        # Create intermediate Vec variables
        vec_vars_p = ["v_out$(i)" for i in 1:n2]
        vec_vars_m = ["v_out$(i+n2)" for i in 1:n2]

        lhs_p = join(vec_vars_p, ", ")
        rhs_p = join(["$(t[i]) + $(t[i+n2])" for i in 1:n2], ", ")

        lhs_m = join(vec_vars_m, ", ")
        rhs_m = join(["$(t[i]) - $(t[i+n2])" for i in 1:n2], ", ")

        code = "$py\n$lhs_p = $rhs_p\n$lhs_m = $rhs_m\n"

        # Combine and store
        if n2 * 2 == floats_per_vec && n2 == 4
            # FFT8 case: combine into wide vectors
            store_pos_p = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos_m = parse(Int, match(r"\[(\d+)\]", y[2*n2 + 1]).captures[1])

            code *= """let
    y_tmp1 = shufflevector($(vec_vars_p[1]), $(vec_vars_p[2]), Val((0, 1, 2, 3)))
    y_tmp2 = shufflevector($(vec_vars_p[3]), $(vec_vars_p[4]), Val((0, 1, 2, 3)))
    y_wide1 = shufflevector(y_tmp1, y_tmp2, Val((0, 1, 2, 3, 4, 5, 6, 7)))
    vstore(y_wide1, $output_buffer, $store_pos_p)
end
let
    y_tmp3 = shufflevector($(vec_vars_m[1]), $(vec_vars_m[2]), Val((0, 1, 2, 3)))
    y_tmp4 = shufflevector($(vec_vars_m[3]), $(vec_vars_m[4]), Val((0, 1, 2, 3)))
    y_wide2 = shufflevector(y_tmp3, y_tmp4, Val((0, 1, 2, 3, 4, 5, 6, 7)))
    vstore(y_wide2, $output_buffer, $store_pos_m)
end
"""
        else
            # Individual stores
            store_positions_p = [parse(Int, match(r"\[(\d+)\]", y[2*i - 1]).captures[1]) for i in 1:n2]
            store_positions_m = [parse(Int, match(r"\[(\d+)\]", y[2*(i+n2) - 1]).captures[1]) for i in 1:n2]

            stores_p = join(["vstore($(vec_vars_p[i]), $output_buffer, $(store_positions_p[i]))" for i in 1:n2], "\n")
            stores_m = join(["vstore($(vec_vars_m[i]), $output_buffer, $(store_positions_m[i]))" for i in 1:n2], "\n")
            code *= "$stores_p\n$stores_m\n"
        end
        return code
    else
        # Non-root: assign to y variables
        lhs_p = join([y[i] for i in 1:n2], ", ")
        rhs_p = join(["$(t[i]) + $(t[i+n2])" for i in 1:n2], ", ")

        lhs_m = join([y[i+n2] for i in 1:n2], ", ")
        rhs_m = join(["$(t[i]) - $(t[i+n2])" for i in 1:n2], ", ")

        return "$lhs_p = $rhs_p\n$lhs_m = $rhs_m\n"
    end
end

# ============================================================================
# **KEY FUNCTION**: Generate butterfly with twiddle fusion
# ============================================================================

function generate_general_butterfly_fused(t, w, y, n2, root, ::Type{T}, floats_per_vec, output_buffer, py) where T
    code_parts = String[]

    # 1. Generate sum/diff temporaries
    if n2 > 1
        for i in 2:n2
            push!(code_parts, "tmp$(i-2) = $(t[i]) + $(t[i+n2])")
        end
        for i in 2:n2
            push!(code_parts, "tmp$(n2+i-3) = $(t[i]) - $(t[i+n2])")
        end
    end

    # 2. Build Op array for FIRST HALF (plus operations)
    # Use temp variable names that will be recombined
    ops_p = Op[]
    if w[1] == "1"
        push!(ops_p, Op("v_out1", "$(t[1]) + $(t[1+n2])", "1"))
    else
        push!(ops_p, Op("v_out1", "$(t[1]) + $(t[1+n2])", w[1]))
    end
    for i in 2:n2
        push!(ops_p, Op("v_out$i", "tmp$(i-2)", w[i]))
    end

    # 3. Build Op array for SECOND HALF (minus operations)
    ops_m = Op[]
    push!(ops_m, Op("v_out$(n2+1)", "$(t[1]) - $(t[1+n2])", w[n2+1]))
    for i in 2:n2
        push!(ops_m, Op("v_out$(n2+i)", "tmp$(n2+i-3)", w[n2+i]))
    end

    # 4. **FUSION MAGIC**: Use merge_ops to fuse twiddle operations
    # merge_ops will generate wide_result for us
    fused_code_p = merge_ops(ops_p, T, 2, 256)  # floats_per_vec=2 (Vec{2,T})
    fused_code_m = merge_ops(ops_m, T, 2, 256)

    push!(code_parts, fused_code_p)
    push!(code_parts, fused_code_m)

    # 5. **AUTO-DETECT AND STORE**
    if root
        # Extract and store wide_result if it exists
        if n2 == 4 && floats_per_vec == 8
            # FFT8 case: Extract from wide_result or combine v_out variables
            store_pos_p = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos_m = parse(Int, match(r"\[(\d+)\]", y[2*n2 + 1]).captures[1])

            # Combine v_out variables into wide vectors for storage
            push!(code_parts, """
# Combine and store wide results
let
    wide_p = shufflevector(v_out1, v_out2, v_out3, v_out4, Val((0,1,2,3,4,5,6,7)))
    wide_m = shufflevector(v_out5, v_out6, v_out7, v_out8, Val((0,1,2,3,4,5,6,7)))
    vstore(wide_p, $output_buffer, $store_pos_p)
    vstore(wide_m, $output_buffer, $store_pos_m)
end
""")
        else
            # Fallback: individual stores
            store_positions_p = [parse(Int, match(r"\[(\d+)\]", y[2*i - 1]).captures[1]) for i in 1:n2]
            store_positions_m = [parse(Int, match(r"\[(\d+)\]", y[2*(i+n2) - 1]).captures[1]) for i in 1:n2]

            for i in 1:n2
                push!(code_parts, "vstore(v_out$i, $output_buffer, $(store_positions_p[i]))")
            end
            for i in 1:n2
                push!(code_parts, "vstore(v_out$(i+n2), $output_buffer, $(store_positions_m[i]))")
            end
        end
    else
        # Non-root: Assign to y variables
        for i in 1:n2
            push!(code_parts, "$(y[i]) = v_out$i")
        end
        for i in 1:n2
            push!(code_parts, "$(y[i+n2]) = v_out$(i+n2)")
        end
    end

    return join(code_parts, "\n")
end

# ============================================================================
# HELPER: Generate DIT butterfly with D-matrix twiddles (similar to above)
# ============================================================================

function generate_dit_butterfly_fused(t, d, y, n2, root, ::Type{T}, floats_per_vec, output_buffer, py) where T
    # Similar to generate_general_butterfly_fused but uses d instead of w
    # Implementation follows same pattern - build Op arrays and call merge_ops
    code_parts = String[]

    # Generate sum/diff temporaries
    if n2 > 1
        for i in 2:n2
            push!(code_parts, "tmp$(i-2) = $(t[i]) + $(t[i+n2])")
        end
        for i in 2:n2
            push!(code_parts, "tmp$(n2+i-3) = $(t[i]) - $(t[i+n2])")
        end
    end

    # Build Op arrays with D-matrix twiddles
    ops_p = Op[]
    push!(ops_p, Op("v_out1", "$(t[1]) + $(t[1+n2])", "1"))
    for i in 2:n2
        push!(ops_p, Op("v_out$i", "tmp$(i-2)", d[i-1]))
    end

    ops_m = Op[]
    push!(ops_m, Op("v_out$(n2+1)", "$(t[1]) - $(t[1+n2])", d[n2]))
    for i in 2:n2
        push!(ops_m, Op("v_out$(n2+i)", "tmp$(n2+i-3)", d[n2+i-1]))
    end

    # Fuse
    fused_code_p = merge_ops(ops_p, T, 2, 256)
    fused_code_m = merge_ops(ops_m, T, 2, 256)

    push!(code_parts, fused_code_p)
    push!(code_parts, fused_code_m)

    # Store (same logic as generate_general_butterfly_fused)
    if root
        if n2 == 4 && floats_per_vec == 8
            store_pos_p = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos_m = parse(Int, match(r"\[(\d+)\]", y[2*n2 + 1]).captures[1])

            push!(code_parts, """
let
    wide_p = shufflevector(v_out1, v_out2, v_out3, v_out4, Val((0,1,2,3,4,5,6,7)))
    wide_m = shufflevector(v_out5, v_out6, v_out7, v_out8, Val((0,1,2,3,4,5,6,7)))
    vstore(wide_p, $output_buffer, $store_pos_p)
    vstore(wide_m, $output_buffer, $store_pos_m)
end
""")
        else
            store_positions_p = [parse(Int, match(r"\[(\d+)\]", y[2*i - 1]).captures[1]) for i in 1:n2]
            store_positions_m = [parse(Int, match(r"\[(\d+)\]", y[2*(i+n2) - 1]).captures[1]) for i in 1:n2]

            for i in 1:n2
                push!(code_parts, "vstore(v_out$i, $output_buffer, $(store_positions_p[i]))")
            end
            for i in 1:n2
                push!(code_parts, "vstore(v_out$(i+n2), $output_buffer, $(store_positions_m[i]))")
            end
        end
    else
        for i in 1:n2
            push!(code_parts, "$(y[i]) = v_out$i")
        end
        for i in 1:n2
            push!(code_parts, "$(y[i+n2]) = v_out$(i+n2)")
        end
    end

    return join(code_parts, "\n")
end
