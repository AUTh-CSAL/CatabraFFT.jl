@enum SuffixFlag::UInt8 begin
    NONE = 0
    VEC = 1
    MAT = 2
    LAYERED = 3
    Y = 4
    IVDEP = 5
end

struct SuffixFlags
    mask::UInt8
    SuffixFlags(mask::UInt8) = new(mask)
end

export SuffixFlag, SuffixFlags
export NONE, VEC, MAT, LAYERED, Y, IVDEP
export empty_flags, flags
export is_empty, has_flag, add_flag, get_active_flags
export flag_to_string, flags_to_strings, flags_to_suffix_string

function _flag_to_bit(flag::SuffixFlag)::UInt8
    if flag == NONE
        return 0x01  # Bit 0
    elseif flag == VEC
        return 0x02  # Bit 1
    elseif flag == MAT
        return 0x04  # Bit 2
    elseif flag == LAYERED
        return 0x08  # Bit 3
    elseif flag == Y
        return 0x10  # Bit 4
    elseif flag == IVDEP
        return 0x20  # Bit 5
    else
        return 0x01
    end
end

function _flags_to_mask(flag_args...)::UInt8
    if isempty(flag_args)
        return 0x01  # Default to NONE
    end
    
    # *** KEY ENFORCEMENT: Check if NONE is present with other flags ***
    has_none = NONE in flag_args
    non_none_flags = filter(f -> f != NONE, flag_args)
    
    if has_none && !isempty(non_none_flags)
        # *** NONE with others: keep only non-NONE flags ***
        return reduce(|, (_flag_to_bit(f) for f in non_none_flags), init=0x00)
    elseif has_none && isempty(non_none_flags)
        # Only NONE
        return 0x01
    else
        # Only non-NONE flags
        mask = reduce(|, (_flag_to_bit(f) for f in non_none_flags), init=0x00)
        return mask == 0x00 ? 0x01 : mask
    end
end

empty_flags() = SuffixFlags(0x01)
flags(args::SuffixFlag...) = SuffixFlags(_flags_to_mask(args...))
is_empty(sf::SuffixFlags) = sf.mask == 0x01
has_flag(sf::SuffixFlags, flag::SuffixFlag) = (sf.mask & _flag_to_bit(flag)) != 0

function add_flag(sf::SuffixFlags, flag::SuffixFlag)
    if flag == NONE
        # *** ADDING NONE CLEARS ALL OTHERS ***
        return empty_flags()
    else
        # *** ADDING REAL FLAG REMOVES NONE ***
        new_mask = sf.mask & ~0x01  # Remove NONE bit
        new_mask |= _flag_to_bit(flag)  # Add new flag
        return SuffixFlags(new_mask)
    end
end

function get_active_flags(sf::SuffixFlags)
    if is_empty(sf)
        return [NONE]
    end
    
    active = SuffixFlag[]
    if has_flag(sf, VEC)
        push!(active, VEC)
    end
    if has_flag(sf, MAT)
        push!(active, MAT)
    end
    if has_flag(sf, LAYERED)
        push!(active, LAYERED)
    end
    if has_flag(sf, Y)
        push!(active, Y)
    end
    if has_flag(sf, IVDEP)
        push!(active, IVDEP)
    end
    return active
end

function show_flags(sf::SuffixFlags, description::String)
    active = get_active_flags(sf)
    mask_binary = string(sf.mask, base=2, pad=5)
    println("$description")
    println("  Mask: 0x$(string(sf.mask, base=16, pad=2)) = $mask_binary")
    println("  Active: $active")
    println("  Is empty: $(is_empty(sf))")
    println()
end

# ===== ENUM TO STRING CONVERSION =====
function flag_to_string(flag::SuffixFlag)::String
    if flag == NONE
        return ""  # NONE doesn't contribute to kernel name
    elseif flag == VEC
        return "vec"
    elseif flag == MAT
        return "mat"
    elseif flag == LAYERED
        return "layered"
    elseif flag == Y
        return "y"
    elseif flag == IVDEP
        return "ivdep"
    else
        return ""
    end
end

function flags_to_strings(suffix_flags::SuffixFlags)::Vector{String}
    if is_empty(suffix_flags)
        return String[]  # Empty array for NONE
    end
    
    active = get_active_flags(suffix_flags)
    return [flag_to_string(flag) for flag in active if flag != NONE]
end

function flags_to_suffix_string(suffix_flags::SuffixFlags)::String
    suffix_strings = flags_to_strings(suffix_flags)
    return join(suffix_strings, "_")
end
