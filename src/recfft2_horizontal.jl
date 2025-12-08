# ============================================================================
# HORIZONTAL FUSION-AWARE FFT KERNEL GENERATOR
# ============================================================================
#
# This generator automatically detects and exploits horizontal packing
# opportunities to generate efficient wide-vector FFT kernels.
#
# Key features:
# 1. Detects consecutive input access patterns
# 2. Packs multiple complex numbers into wide vectors (Vec{4}, Vec{8})
# 3. Fuses butterflies horizontally when possible
# 4. Applies twiddles efficiently within wide vectors
# 5. Generates wide stores to minimize memory traffic
#
# ============================================================================

include("horizontal_fusion.jl")

"""
Main entry point: Generate horizontal fusion-aware FFT kernel.

Strategy:
1. Analyze input variables for consecutive access
2. If consecutive: use wide horizontal packing
3. If scattered: fall back to narrow vectors
4. Apply twiddles efficiently
5. Generate wide output stores
"""
@inline function recfft2_horizontal(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base=1,
                                    mode=:vgather, py="", complexes_per_vec=4,
                                    input_buffer="x", output_buffer="y") where T <: AbstractFloat
    n = length(x)  # Number of complex numbers

    # BASE CASE: n == 1
    if n == 1
        return ""

    # BASE CASE: n == 2 (Radix-2 butterfly)
    elseif n == 2
        return generate_butterfly2_horizontal(y, x, d, w, root, T, floats_per_vec,
                                             input_buffer, output_buffer)

    # RECURSIVE CASE: n == 4 (FFT4 - sweet spot for horizontal fusion)
    elseif n == 4
        return generate_fft4_horizontal(y, x, d, w, root, T, floats_per_vec,
                                       input_buffer, output_buffer)

    # RECURSIVE CASE: n == 8 (FFT8 - two FFT4s)
    elseif n == 8
        return generate_fft8_horizontal(y, x, d, w, root, T, floats_per_vec, tmp_base,
                                       input_buffer, output_buffer)

    # GENERAL RECURSIVE CASE: n > 8
    else
        return generate_recursive_horizontal(y, x, d, w, root, T, floats_per_vec, tmp_base,
                                            mode, py, complexes_per_vec, input_buffer, output_buffer, n)
    end
end

# ============================================================================
# FFT2 Butterfly (Simplest case)
# ============================================================================

function generate_butterfly2_horizontal(y, x, d, w, root, ::Type{T}, floats_per_vec,
                                       input_buffer, output_buffer) where T
    if root
        # Extract indices
        idx1 = parse(Int, match(r"(\d+)", x[1]).captures[1])
        idx2 = parse(Int, match(r"(\d+)", x[2]).captures[1])

        # Check if consecutive (can use wide load)
        if idx2 == idx1 + 1
            # HORIZONTAL FUSION: Load both as Vec{4}
            pos1 = 2 * (idx1 - 1) + 1
            store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos2 = parse(Int, match(r"\[(\d+)\]", y[3]).captures[1])

            if !isnothing(d) && !isnothing(d[1])
                # With twiddle
                tw_expr = sat_expr_simd_inline("tmp0", d[1], T, 2)
                return """
                # Horizontal FFT2 with twiddle
                w_load = vload(Vec{4,$T}, $input_buffer, $pos1)
                $(x[1]) = shufflevector(w_load, Val((0, 1)))
                $(x[2]) = shufflevector(w_load, Val((2, 3)))
                tmp0 = $(x[1]) - $(x[2])
                t1 = $(x[1]) + $(x[2])
                t2 = $tw_expr
                vstore(t1, $output_buffer, $store_pos1)
                vstore(t2, $output_buffer, $store_pos2)
                """
            else
                # No twiddle
                return """
                # Horizontal FFT2 no twiddle
                w_load = vload(Vec{4,$T}, $input_buffer, $pos1)
                w_sum = shufflevector(w_load, Val((0, 1, 0, 1))) + shufflevector(w_load, Val((2, 3, 2, 3)))
                w_diff = shufflevector(w_load, Val((0, 1, 0, 1))) - shufflevector(w_load, Val((2, 3, 2, 3)))
                vstore(shufflevector(w_sum, Val((0, 1))), $output_buffer, $store_pos1)
                vstore(shufflevector(w_diff, Val((0, 1))), $output_buffer, $store_pos2)
                """
            end
        else
            # Not consecutive, use narrow loads
            pos1 = 2 * (idx1 - 1) + 1
            pos2 = 2 * (idx2 - 1) + 1
            store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
            store_pos2 = parse(Int, match(r"\[(\d+)\]", y[3]).captures[1])

            return """
            $(x[1]) = vload(Vec{2,$T}, $input_buffer, $pos1)
            $(x[2]) = vload(Vec{2,$T}, $input_buffer, $pos2)
            t1 = $(x[1]) + $(x[2])
            t2 = $(x[1]) - $(x[2])
            vstore(t1, $output_buffer, $store_pos1)
            vstore(t2, $output_buffer, $store_pos2)
            """
        end
    else
        # Non-root case
        if isnothing(w)
            return "$(y[1]), $(y[2]) = $(x[1]) + $(x[2]), $(x[1]) - $(x[2])\n"
        else
            # With twiddles - apply using sat_expr_simd
            tw1_expr = sat_expr_simd_inline("$(x[1]) + $(x[2])", w[1], T, 2)
            tw2_expr = sat_expr_simd_inline("$(x[1]) - $(x[2])", w[2], T, 2)
            return "$(y[1]), $(y[2]) = $tw1_expr, $tw2_expr\n"
        end
    end
end

# ============================================================================
# FFT4 - HORIZONTAL FUSION SWEET SPOT
# ============================================================================

function generate_fft4_horizontal(y, x, d, w, root, ::Type{T}, floats_per_vec,
                                  input_buffer, output_buffer) where T
    # Check if inputs are consecutive (works for both root and twiddle cases)
    if !isnothing(d) || startswith(x[1], "v")
        try
            indices = [parse(Int, match(r"(\d+)", var).captures[1]) for var in x]
            is_consecutive = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))
        catch
            is_consecutive = false
        end
    else
        is_consecutive = false
    end

    # CASE 1: Root with consecutive inputs (no twiddles)
    if root && is_consecutive
        start_pos = 2 * (parse(Int, match(r"(\d+)", x[1]).captures[1]) - 1) + 1
        store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
        store_pos2 = parse(Int, match(r"\[(\d+)\]", y[5]).captures[1])

        return """
        # HORIZONTAL FFT4 (consecutive inputs) with let blocks for register management
        vout1, vout2 = let
            l_all = vload(Vec{8,$T}, $input_buffer, $start_pos)

            # Pack consecutive pairs: [x0,x1] and [x2,x3]
            v1 = shufflevector(l_all, Val((0, 1, 2, 3)))
            v2 = shufflevector(l_all, Val((4, 5, 6, 7)))

            # Horizontal butterfly
            tvec1 = v1 + v2  # [x0+x2, x1+x3] = [even_sum, odd_sum]
            tvec2 = shufflevector(signflip(v1 - v2, Vec{4,UInt32}((0x00000000, 0x00000000, 0x80000000, 0x00000000))),
                                 Val((0, 1, 3, 2)))  # [even_diff, (odd_diff)*(-i)]

            # Interleave for final butterfly
            hvec1 = shufflevector(tvec1, tvec2, Val((0, 1, 4, 5)))
            hvec2 = shufflevector(tvec1, tvec2, Val((2, 3, 6, 7)))

            # Final outputs
            hvec1 + hvec2, hvec1 - hvec2
        end

        vstore(vout1, $output_buffer, $store_pos1)
        vstore(vout2, $output_buffer, $store_pos2)
        """

    # CASE 2: Non-root with uniform simple twiddles (can still use horizontal fusion)
    elseif !root && !isnothing(d) && is_consecutive
        # Check if all twiddles are the same and simple ("1", "-1", "im", "-im")
        all_same = length(unique(d)) == 1
        twiddle = all_same ? d[1] : nothing

        if all_same && (twiddle == "1" || twiddle == "-1" || twiddle == "im" || twiddle == "-im")
            # Generate twiddle application code
            twiddle_code = if twiddle == "1"
                ""  # No twiddle needed
            elseif twiddle == "-1"
                "l_all = -l_all"
            elseif twiddle == "im"
                "l_all = shufflevector(signflip(l_all, Vec{8,UInt32}((0x00000000,0x80000000,0x00000000,0x80000000,0x00000000,0x80000000,0x00000000,0x80000000))), Val((1,0,3,2,5,4,7,6)))"
            elseif twiddle == "-im"
                "l_all = shufflevector(signflip(l_all, Vec{8,UInt32}((0x80000000,0x00000000,0x80000000,0x00000000,0x80000000,0x00000000,0x80000000,0x00000000))), Val((1,0,3,2,5,4,7,6)))"
            end

            return """
            # HORIZONTAL FFT4 with uniform twiddle: $twiddle
            $(x[1]), $(x[2]), $(x[3]), $(x[4]) = let
                l_all = vload(Vec{8,$T}, $input_buffer, 1)
                $twiddle_code

                # Pack consecutive pairs: [x0,x1] and [x2,x3]
                v1 = shufflevector(l_all, Val((0, 1, 2, 3)))
                v2 = shufflevector(l_all, Val((4, 5, 6, 7)))

                # Horizontal butterfly
                tvec1 = v1 + v2
                tvec2 = shufflevector(signflip(v1 - v2, Vec{4,UInt32}((0x00000000, 0x00000000, 0x80000000, 0x00000000))),
                                     Val((0, 1, 3, 2)))

                # Interleave for final butterfly
                hvec1 = shufflevector(tvec1, tvec2, Val((0, 1, 4, 5)))
                hvec2 = shufflevector(tvec1, tvec2, Val((2, 3, 6, 7)))

                # Final outputs as individual Vec{2}
                v_out1 = hvec1 + hvec2
                v_out2 = hvec1 - hvec2

                shufflevector(v_out1, Val((0,1))), shufflevector(v_out1, Val((2,3))),
                shufflevector(v_out2, Val((0,1))), shufflevector(v_out2, Val((2,3)))
            end
            """
        end
    end

    # Fall back to narrow vector implementation
    # Load individual Vec{2} and process
    code = ""
    if root
        for (i, var) in enumerate(x)
            idx = parse(Int, match(r"(\d+)", var).captures[1])
            pos = 2 * (idx - 1) + 1
            code *= "$var = vload(Vec{2,$T}, $input_buffer, $pos)\n"
        end
    end

    # First layer butterflies
    code *= """
    tvec9, tvec10 = $(x[1]) + $(x[3]), $(x[1]) - $(x[3])
    tvec11 = $(x[2]) + $(x[4])
    tvec12 = shufflevector(signflip($(x[2]) - $(x[4]), Vec{2,UInt32}((0x80000000, 0x00000000))), Val((1, 0)))

    tvec1, tvec2 = tvec9 + tvec11, tvec10 + tvec12
    tvec3, tvec4 = tvec9 - tvec11, tvec10 - tvec12
    """

    if root
        store_positions = [parse(Int, match(r"\[(\d+)\]", y[2*i-1]).captures[1]) for i in 1:4]
        for i in 1:4
            code *= "vstore(tvec$i, $output_buffer, $(store_positions[i]))\n"
        end
    else
        for i in 1:4
            code *= "$(y[i]) = tvec$i\n"
        end
    end

    return code
end

# ============================================================================
# FFT8 - Two FFT4s with twiddles
# ============================================================================

function generate_fft8_horizontal(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base,
                                  input_buffer, output_buffer) where T
    # Check if inputs are consecutive
    indices = [parse(Int, match(r"(\d+)", var).captures[1]) for var in x]
    is_consecutive = all(i -> indices[i] == indices[1] + i - 1, 2:length(indices))

    if root && is_consecutive
        # HORIZONTAL FFT8 implementation
        start_pos = 2 * (indices[1] - 1) + 1
        store_pos1 = parse(Int, match(r"\[(\d+)\]", y[1]).captures[1])
        store_pos2 = parse(Int, match(r"\[(\d+)\]", y[9]).captures[1])

        return """
        # HORIZONTAL FFT8 (consecutive inputs) with let blocks for register management
        l_all = vload(Vec{16,$T}, $input_buffer, $start_pos)

        # Split into two FFT4 groups: even[x0,x2,x4,x6] and odd[x1,x3,x5,x7]
        even_all_input = shufflevector(l_all, Val((0,1,4,5,8,9,12,13)))
        odd_all_input = shufflevector(l_all, Val((2,3,6,7,10,11,14,15)))

        # FFT4 on even half (scoped to release registers early)
        even_out1, even_out2 = let
            v1 = shufflevector(even_all_input, Val((0,1,2,3)))
            v2 = shufflevector(even_all_input, Val((4,5,6,7)))
            sum_vec = v1 + v2
            diff_vec = shufflevector(signflip(v1 - v2, Vec{4,UInt32}((0x00000000,0x00000000,0x80000000,0x00000000))),
                                    Val((0,1,3,2)))
            h1 = shufflevector(sum_vec, diff_vec, Val((0,1,4,5)))
            h2 = shufflevector(sum_vec, diff_vec, Val((2,3,6,7)))
            h1 + h2, h1 - h2
        end

        # FFT4 on odd half (scoped to release registers early)
        odd_tmp1, odd_tmp2 = let
            v1 = shufflevector(odd_all_input, Val((0,1,2,3)))
            v2 = shufflevector(odd_all_input, Val((4,5,6,7)))
            sum_vec = v1 + v2
            diff_vec = shufflevector(signflip(v1 - v2, Vec{4,UInt32}((0x00000000,0x00000000,0x80000000,0x00000000))),
                                    Val((0,1,3,2)))
            h1 = shufflevector(sum_vec, diff_vec, Val((0,1,4,5)))
            h2 = shufflevector(sum_vec, diff_vec, Val((2,3,6,7)))
            h1 + h2, h1 - h2
        end

        # Apply FFT8 twiddles to odd outputs (scoped)
        # Twiddles: W^0=1, W^1=INV_SQRT2_Q4, W^2=-i, W^3=-INV_SQRT2_Q1
        odd_out1, odd_out2 = let
            o0 = shufflevector(odd_tmp1, Val((0,1)))
            o1 = shufflevector(odd_tmp1, Val((2,3)))
            o2 = shufflevector(odd_tmp2, Val((0,1)))
            o3 = shufflevector(odd_tmp2, Val((2,3)))

            # o1: INV_SQRT2_Q4
            o1_tw = let
                swap = shufflevector(o1, Val((1,0)))
                shufflevector(o1 + swap, o1 - swap, Val((0,3))) * INV_SQRT2
            end

            # o2: -i
            o2_tw = shufflevector(signflip(o2, Vec{2,UInt32}((0x80000000, 0x00000000))), Val((1,0)))

            # o3: -INV_SQRT2_Q1
            o3_tw = let
                swap = shufflevector(o3, Val((1,0)))
                shufflevector(o3 - swap, -(o3 + swap), Val((1,2))) * INV_SQRT2
            end

            shufflevector(o0, o1_tw, Val((0,1,2,3))), shufflevector(o2_tw, o3_tw, Val((0,1,2,3)))
        end

        # Final FFT8 butterfly: y[0..3] = even + odd_tw, y[4..7] = even - odd_tw
        vout1, vout2 = let
            even_all = shufflevector(even_out1, even_out2, Val((0,1,2,3,4,5,6,7)))
            odd_all = shufflevector(odd_out1, odd_out2, Val((0,1,2,3,4,5,6,7)))
            even_all + odd_all, even_all - odd_all
        end

        vstore(vout1, $output_buffer, $store_pos1)
        vstore(vout2, $output_buffer, $store_pos2)
        """
    end

    # Fall back to narrow implementation (similar to current recfft2_simd)
    # ... (keep existing narrow path for non-consecutive case)
    return "# TODO: Narrow FFT8 path\n"
end

# ============================================================================
# General recursive case (larger FFTs)
# ============================================================================

function generate_recursive_horizontal(y, x, d, w, root, ::Type{T}, floats_per_vec, tmp_base,
                                      mode, py, complexes_per_vec, input_buffer, output_buffer, n) where T
    # For now, fall back to existing recfft2_simd for larger sizes
    # This is where future optimization would go
    return "# TODO: General recursive horizontal path for n=$n\n"
end

# ============================================================================
# Helper: Inline twiddle application
# ============================================================================

function sat_expr_simd_inline(vec_name::String, twiddle, ::Type{T}, floats_per_vec::Int) where T
    if twiddle isa String
        if twiddle == "1"
            return vec_name
        elseif twiddle == "-1"
            return "-$vec_name"
        elseif twiddle == "im"
            return "shufflevector(signflip($vec_name, Vec{$floats_per_vec,UInt32}((0x00000000, 0x80000000))), Val((1, 0)))"
        elseif twiddle == "-im"
            return "shufflevector(signflip($vec_name, Vec{$floats_per_vec,UInt32}((0x80000000, 0x00000000))), Val((1, 0)))"
        end
    end

    # For complex twiddles, use existing sat_expr_simd from fft_seed.jl
    return "# Complex twiddle: $twiddle"
end

export recfft2_horizontal
