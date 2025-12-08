# ============================================================================
# HORIZONTAL SIMD FUSION ENGINE
# ============================================================================
#
# Philosophy: Pack CONSECUTIVE complex numbers into WIDE vectors
# Example: Instead of 4× Vec{2,Float32}, use 1× Vec{8,Float32}
#
# Key Innovation: Horizontal butterfly fusion
#   v1 = [x0, x1]  (consecutive pair)
#   v2 = [x2, x3]  (consecutive pair)
#   sum = v1 + v2 = [x0+x2, x1+x3] = [even_sum, odd_sum]  ← TWO DFTs at once!
#
# ============================================================================

"""
HorizontalGroup represents a group of complex operations that can be packed together.
"""
struct HorizontalGroup
    vars::Vector{String}           # Variable names (e.g., ["v1", "v2", "v3", "v4"])
    start_index::Int              # Starting complex index (0-based)
    width::Int                    # Number of complex numbers (2, 4, etc.)
    twiddles::Vector{Any}         # Twiddle factor for each complex
end

"""
Analyze variable names to detect consecutive access patterns.
Returns groups that can be packed horizontally.
"""
function analyze_horizontal_groups(vars::Vector{String}, twiddles::Vector{Any}, max_width::Int)::Vector{HorizontalGroup}
    if isempty(vars)
        return HorizontalGroup[]
    end

    # Extract indices from variable names (e.g., "v1" → 1, "v5" → 5)
    indices = Int[]
    for var in vars
        m = match(r"v?(\d+)", var)
        if !isnothing(m)
            push!(indices, parse(Int, m.captures[1]))
        else
            # Can't parse index, treat as non-groupable
            return HorizontalGroup[]
        end
    end

    # Check if indices are consecutive starting from some base
    if length(indices) < 2
        return HorizontalGroup[]
    end

    # Find consecutive runs
    groups = HorizontalGroup[]
    i = 1
    while i <= length(indices)
        # Try to build a consecutive group starting at i
        group_start = i
        group_end = i

        # Extend while consecutive and within max_width
        while group_end < length(indices) &&
              indices[group_end + 1] == indices[group_end] + 1 &&
              (group_end - group_start + 1) < max_width
            group_end += 1
        end

        # Only create group if we have at least 2 consecutive
        if group_end > group_start
            width = group_end - group_start + 1
            push!(groups, HorizontalGroup(
                vars[group_start:group_end],
                indices[group_start] - 1,  # Convert to 0-based
                width,
                twiddles[group_start:group_end]
            ))
            i = group_end + 1
        else
            i += 1
        end
    end

    return groups
end

"""
Generate wide vector load for a horizontal group.
"""
function gen_wide_load(group::HorizontalGroup, ::Type{T}, input_buffer::String) where T <: AbstractFloat
    n_floats = 2 * group.width
    start_pos = 2 * group.start_index + 1  # 1-based position

    wide_var = "w$(group.start_index)"
    code = "$wide_var = vload(Vec{$n_floats,$T}, $input_buffer, $start_pos)\n"

    # Extract individual complex numbers if needed later
    for (i, var) in enumerate(group.vars)
        offset = 2 * (i - 1)
        code *= "$var = shufflevector($wide_var, Val(($(offset), $(offset+1))))\n"
    end

    return (code, wide_var)
end

"""
Generate horizontal butterfly for a group.
Packs consecutive butterflies into single wide operation.
"""
function gen_horizontal_butterfly(group::HorizontalGroup, ::Type{T}) where T
    if group.width < 2
        return nothing  # Need at least 2 for horizontal fusion
    end

    # Check if group represents a proper FFT butterfly structure
    # For FFT4: need 4 consecutive inputs that split into even=[0,2] odd=[1,3]
    # For horizontal packing: [x0,x1] and [x2,x3]

    if group.width == 2
        # Simple 2-way butterfly
        v1, v2 = group.vars
        return """
        # Horizontal 2-way butterfly
        h_sum = $v1 + $v2
        h_diff = $v1 - $v2
        """
    elseif group.width == 4
        # 4-way horizontal butterfly (FFT4 pattern)
        # Pack as: [v1,v2] and [v3,v4] where indices are consecutive
        v1, v2, v3, v4 = group.vars
        return """
        # Horizontal 4-way butterfly for FFT4
        # Pack consecutive pairs: [$(v1),$(v2)] and [$(v3),$(v4)]
        w_lo = shufflevector($v1, $v2, Val((0, 1, 2, 3)))
        w_hi = shufflevector($v3, $v4, Val((0, 1, 2, 3)))
        h_sum = w_lo + w_hi      # [$(v1)+$(v3), $(v2)+$(v4)]
        h_diff = w_lo - w_hi     # [$(v1)-$(v3), $(v2)-$(v4)]
        """
    end

    return nothing
end

"""
Apply twiddles to wide vector with selective masking.
Key insight: Different complexes in same vector can have different twiddles!
"""
function gen_wide_twiddle(wide_var::String, twiddles::Vector{Any}, ::Type{T}, width::Int) where T
    # Group twiddles by type
    all_identity = all(tw -> tw == "1" || isnothing(tw), twiddles)
    all_same = length(unique(twiddles)) == 1

    if all_identity
        return "$wide_var  # No twiddle needed"
    end

    if all_same && twiddles[1] == "-im"
        # All complexes get same -im twiddle - can do in wide vector
        n_floats = 2 * width
        mask_bits = String[]
        for i in 1:width
            push!(mask_bits, "0x80000000", "0x00000000")  # Flip real of each complex
        end

        return """shufflevector(signflip($wide_var, Vec{$n_floats,UInt32}(($(join(mask_bits, ", "))))),
                                  Val(($(join([i for pair in [[2i-1, 2i-2] for i in 1:width] for i in pair], ", ")))
                                ))"""
    end

    # Heterogeneous twiddles - need per-complex handling
    # This is where it gets tricky: apply different twiddles to different positions

    # For now, extract and apply individually (optimization opportunity)
    code_parts = String[]
    for (i, tw) in enumerate(twiddles)
        offset = 2 * (i - 1)
        tmp_var = "tw_tmp$i"
        push!(code_parts, "$tmp_var = shufflevector($wide_var, Val(($offset, $(offset+1))))")

        if tw == "1" || isnothing(tw)
            # Identity, keep as-is
        elseif tw == "-im"
            push!(code_parts, "$tmp_var = shufflevector(signflip($tmp_var, Vec{2,UInt32}((0x80000000, 0x00000000))), Val((1, 0)))")
        elseif tw == "im"
            push!(code_parts, "$tmp_var = shufflevector(signflip($tmp_var, Vec{2,UInt32}((0x00000000, 0x80000000))), Val((1, 0)))")
        # Add more twiddle cases as needed
        end
    end

    # Recombine into wide vector
    if length(twiddles) == 2
        push!(code_parts, "shufflevector(tw_tmp1, tw_tmp2, Val((0, 1, 2, 3)))")
    elseif length(twiddles) == 4
        push!(code_parts, "shufflevector(tw_tmp1, tw_tmp2, tw_tmp3, tw_tmp4, Val((0,1,2,3,4,5,6,7)))")
    end

    return join(code_parts, "\n")
end

"""
Generate wide store for multiple consecutive complex numbers.
"""
function gen_wide_store(vars::Vector{String}, y_base::Int, ::Type{T}, output_buffer::String, width::Int) where T
    n_floats = 2 * width
    store_pos = 2 * y_base + 1

    if width == 2
        return "vstore(shufflevector($(vars[1]), $(vars[2]), Val((0,1,2,3))), $output_buffer, $store_pos)"
    elseif width == 4
        return """let
    w_pair1 = shufflevector($(vars[1]), $(vars[2]), Val((0,1,2,3)))
    w_pair2 = shufflevector($(vars[3]), $(vars[4]), Val((0,1,2,3)))
    w_all = shufflevector(w_pair1, w_pair2, Val((0,1,2,3,4,5,6,7)))
    vstore(w_all, $output_buffer, $store_pos)
end"""
    end

    # General case
    parts = String[]
    for i in 1:2:width
        if i+1 <= width
            push!(parts, "shufflevector($(vars[i]), $(vars[i+1]), Val((0,1,2,3)))")
        end
    end

    if length(parts) == 1
        return "vstore($(parts[1]), $output_buffer, $store_pos)"
    else
        # Hierarchical merging
        return "vstore(shufflevector($(join(parts, ", ")), Val(($(join(0:n_floats-1, ", "))))), $output_buffer, $store_pos)"
    end
end

export HorizontalGroup, analyze_horizontal_groups, gen_wide_load, gen_horizontal_butterfly, gen_wide_twiddle, gen_wide_store
