module RadixGenerator

include("helper_tools.jl")
include("radix_plan.jl")
include("suffix.jl")
include("fft_seed.jl")

using LoopVectorization
using .Radix_Plan
using Libdl

export create_kernel_dictionary, extract_plan_data

# Intel CPU detection and IVM support
const USE_IVM = let
    try
        if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
            cpuinfo = read(`lscpu`, String)
            occursin("Intel", cpuinfo) || occursin("GenuineIntel", cpuinfo)
        else
            false
        end
    catch
        false
    end
end

if USE_IVM
    try
        using IntelVectorMath   # IVM alias exported;
        IVM = IntelVectorMath
        @info "IntelVectorMath.jl enabled for accelerated twiddle factor generation"
    catch
        @warn "IntelVectorMath.jl not available, using Base.cispi"
    end
end

@inline function fast_cispi(phases::Vector{T}) where T <: AbstractFloat
    if USE_IVM && T == Float64
        # IntelVectorMath.cis requires properly signed zeros
        # Multiply first to get phases in radians, then ensure no -0.0
        radian_phases = phases .* T(π)
        @inbounds for i in eachindex(radian_phases)
            if radian_phases[i] == -zero(T)
                radian_phases[i] = zero(T)
            end
        end
        return IVM.cis(radian_phases)
    elseif USE_IVM && T == Float32
        # Convert to Float64, handle -0.0, then convert back
        radian_phases = Float64.(phases) .* π
        @inbounds for i in eachindex(radian_phases)
            if radian_phases[i] == -zero(Float64)
                radian_phases[i] = zero(Float64)
            end
        end
        return ComplexF32.(IVM.cis(radian_phases))
    else
        return cispi.(phases)
    end
end

@inline function fast_cossinpi(phases::Vector{T}, ::Type{T}) where T <: AbstractFloat
    n = length(phases)
    cos_vals = Vector{T}(undef, n)
    sin_vals = Vector{T}(undef, n)

    if USE_IVM && T == Float64
        # Convert phases to radians: phase * π
        radian_phases = phases .* T(π)
        @inbounds for i in eachindex(radian_phases)
            if radian_phases[i] == -zero(T)
                radian_phases[i] = zero(T)
            end
        end

        # Use IVM for vectorized computation
        IVM.sincos!(sin_vals, cos_vals, radian_phases)

    elseif USE_IVM && T == Float32
        # Convert to Float64 for IVM, then convert back
        radian_phases = Float64.(phases) .* π
        @inbounds for i in eachindex(radian_phases)
            if radian_phases[i] == -zero(Float64)
                radian_phases[i] = zero(Float64)
            end
        end

        cos_vals_64 = Vector{Float64}(undef, n)
        sin_vals_64 = Vector{Float64}(undef, n)
        IVM.sincos!(sin_vals_64, cos_vals_64, radian_phases)

        # Convert back to Float32
        @inbounds for i in 1:n
            cos_vals[i] = T(cos_vals_64[i])
            sin_vals[i] = T(sin_vals_64[i])
        end
    else
        # Scalar fallback
        @inbounds for i in 1:n
            cos_vals[i] = cospi(phases[i])
            sin_vals[i] = sinpi(phases[i])
        end
    end

    return cos_vals, sin_vals
end

function create_kernel_dictionary(plan_data::NamedTuple, ::Type{T})::Dict{String, Expr} where T <: AbstractFloat
    kernels = Dict{String, Expr}()
    
    custom_combinations = empty_flags()
    if length(plan_data.operations) != 1 
        custom_combinations = add_flag(custom_combinations, MAT) 
    end
    
    kernel_codes = generate_all_kernel_expressions(plan_data, T; suffix_combinations=custom_combinations)
    
    @inbounds for (name, code) in kernel_codes
        kernels[name] = Expr(:block, code)
    end
    
    return kernels
end

function generate_local_constants_dict(n::Int, ::Type{T}) where T <: AbstractFloat
    @assert ispow2(n) "n must be a power of 2"
    #@assert n >= 8 "n must be at least 8"

    # Collect all unique reduced fractions needed for all sizes from 8 up to n
    fractions = Vector{Tuple{Int,Int}}()

    # Pre-estimate size: for powers of 2, approximately log2(n/4) fractions per level
    estimated_size = max(1, (n >> 3))  # Rough estimate
    sizehint!(fractions, estimated_size)

    # Start from n=8 and work up to the target n
    current_n = 8
    @inbounds while current_n <= n
        denominator = current_n ÷ 2  # For n=8: den=4, n=16: den=8, n=32: den=16, etc.
        max_numerator = denominator ÷ 4  # Only fractions < 1/4 (angles < π/4)

        # Add all odd numerators k where k <= den/4 and gcd(k, denominator) = 1
        # This ensures we only add new fractions in reduced form with angles < π/4
        # Avoids duplicates since cos(θ) = sin(π/2 - θ) for θ < π/4
        for k in 1:2:max_numerator  # Only odd k: 1, 3, 5, 7, ...
            if gcd(k, denominator) == 1
                push!(fractions, (k, denominator))
            end
        end

        current_n <<= 1  # Double for next iteration
    end

    # Pre-allocate dictionary: 1 for INV_SQRT2 + 2 per remaining fraction (cos + sin)
    num_fractions = length(fractions)
    has_inv_sqrt2 = num_fractions > 0 && fractions[1] == (1, 4)
    dict_size = (has_inv_sqrt2 ? 1 : 0) + 2 * (num_fractions - (has_inv_sqrt2 ? 1 : 0))
    constants_dict = Dict{Symbol, T}()
    sizehint!(constants_dict, dict_size)

    # Vectorized trig computation for better performance
    if num_fractions > 1 || (num_fractions == 1 && !has_inv_sqrt2)
        # Build phase array for vectorized computation
        start_idx = has_inv_sqrt2 ? 2 : 1
        num_trig = num_fractions - (has_inv_sqrt2 ? 1 : 0)

        if num_trig > 0
            phases = Vector{T}(undef, num_trig)
            @inbounds for i in 1:num_trig
                num, den = fractions[start_idx + i - 1]
                phases[i] = T(num) / T(den)
            end

            # Compute cos and sin in parallel using vectorized operations
            cos_vals, sin_vals = fast_cossinpi(phases, T)

            # Populate dictionary
            @inbounds for i in 1:num_trig
                num, den = fractions[start_idx + i - 1]
                constants_dict[Symbol("COSPI_$(num)_$(den)")] = cos_vals[i]
                constants_dict[Symbol("SINPI_$(num)_$(den)")] = sin_vals[i]
            end
        end
    end

    # Special case: INV_SQRT2 for 1/4
    if has_inv_sqrt2
        constants_dict[:INV_SQRT2] = T(1/√2)
    end

    return constants_dict
end

function generate_all_kernel_expressions(plan_data::NamedTuple, ::Type{T}; 
                                        suffix_combinations::Union{Nothing, SuffixFlags}=nothing) where T <: AbstractFloat
    kernels = Dict{String, Expr}()
    
    if has_flag(suffix_combinations, NONE)
        op = plan_data.operations[1]
        radix = get_radix_divisor(op.op_type)
        name, expr = generate_kernel_expression(radix, op, suffix_combinations, 0, Vector{T}, true, T)
        kernels[name] = expr
        
    elseif has_flag(suffix_combinations, MAT)
        @inbounds for (stage_idx, op) in enumerate(plan_data.operations)
            radix = get_radix_divisor(op.op_type)
            is_final = (stage_idx == length(plan_data.operations))
            
            SIZE = op.n_groups * op.stride
            n_kernels_needed = SIZE ÷ radix
            
            if !is_final
                next_op = plan_data.operations[stage_idx + 1]
                
                for p in 0:(n_kernels_needed-1)
                    base = (p ÷ op.stride) * (op.stride * radix) + (p % op.stride)
                    output_group = base ÷ next_op.stride
                    d_column_idx = (output_group % next_op.n_groups) + 1
                    
                    D = generate_D_kernel(d_column_idx, radix, op.stride, next_op.stride, next_op.n_groups, T)
                    name, expr = generate_kernel_expression(radix, op, suffix_combinations, p, D, false, T) # Generate appropriate sub-kernel by embedding D matrix layer on it at compile-time!
                    kernels[name] = expr
                end
            else
                vec_suffix = add_flag(suffix_combinations, VEC)
                println("Creating final terminal VEC kernel: $op")
                name, expr = generate_kernel_expression(radix, op, vec_suffix, 0, Vector{T}, true, T)
                kernels[name] = expr
            end
        end
    end
    
    return kernels
end

@inline function generate_kernel_expression(radix::Int, op, suffixes::SuffixFlags, p::Int, D::Union{Vector{Union{String, Twiddle}}, Vector{T}, Type{<:AbstractVector}}, is_last::Bool, ::Type{T}) where T <: AbstractFloat
    if op.eo && is_last
        suffixes = add_flag(suffixes, Y)
    end
    
    name = generate_kernel_name(radix, suffixes, p, op)
    SIZE = op.n_groups * op.stride
    SIMD_BITS = 256
    kernel_body = makefftradix(radix, suffixes, D, p, op, SIZE, T, SIMD_BITS)
    
    return name, kernel_body
end

function generate_kernel_name(radix::Int, suffix_flags::SuffixFlags, p::Int, op)
    has_mat = has_flag(suffix_flags, MAT)
    has_layered = has_flag(suffix_flags, LAYERED)
    
    if has_mat && !has_layered 
        return "fft$(radix)_$(op.stride)x$(op.n_groups)_$(p)!"
    end
    
    base = "fft$(radix)_shell"
    if is_empty(suffix_flags)
        return string(base, "!")
    else
        active_flags = get_active_flags(suffix_flags)
        suffix_parts = [flag_to_string(flag) for flag in active_flags if flag != NONE]
        suffix = join(suffix_parts, "_")
        return string(base, "_", suffix, "!")
    end
end

function generate_D_kernel(p, radix::Int, current_stride::Int, next_stride::Int, n_groups::Int, ::Type{T}) where T <: AbstractFloat
    (next_stride == 1 || n_groups == 1 || p == 1) && return Union{String, Twiddle}[]

    N = next_stride * n_groups
    j = p - 1  # Convert to 0-indexed group

    # Compute effective k values for analytical twiddle generation
    # Each twiddle is cispi(-2 * k * current_stride * j / N)
    # which equals cispi(-2 * effective_k / N) where effective_k = k * current_stride * j
    ks = [k * current_stride * j for k in 1:(radix-1)]

    # Use analytical twiddle expression generator (no trig computation!)
    return get_twiddle_expression(ks, N; T=T)
end

# Helper function to classify twiddle factors analytically
# Returns the appropriate string or Twiddle representation for cispi(-2k/n)
# Constrains numerator to be <= denominator/4 to match generate_local_constants_dict
@inline function classify_twiddle_factor(k_norm::Int, n::Int)
    # w_k = cispi(-2k/n) where k_norm is already normalized to [0, n)

    # Reduce the fraction 2k/n to lowest terms
    numerator = 2 * k_norm
    g = gcd(numerator, n)
    num = numerator ÷ g
    den = n ÷ g

    # Now we have cispi(-num/den)
    # Normalize to [0, 2) by taking mod 2
    phase_times_den = mod(-num, 2*den)  # This gives (-num mod 2*den)

    # Reduce this fraction to lowest terms
    g2 = gcd(phase_times_den, den)
    phase_num = phase_times_den ÷ g2
    phase_den = den ÷ g2

    # Now phase = phase_num/phase_den ∈ [0, 2)
    # We want to express cispi(phase_num/phase_den) using only base twiddles
    # where the numerator is <= denominator/4

    # Use transformations to reduce to canonical form:
    # cispi(x + 1/2) = i * cispi(x)
    # cispi(x + 1) = -cispi(x)
    # cispi(x + 3/2) = -i * cispi(x)

    # Classify based on which octant phase falls into:
    # [0, 1/4]: Q1,    [1/4, 1/2]: ImQ4,   [1/2, 3/4]: ImQ1,    [3/4, 1]: NegQ4
    # [1, 5/4]: NegQ1, [5/4, 3/2]: NegImQ4, [3/2, 7/4]: NegImQ1, [7/4, 2]: Q4

    # Check boundaries using integer arithmetic to avoid floating point
    # phase < 1/4 iff 4*phase_num < phase_den
    if 4 * phase_num <= phase_den
        # phase ∈ [0, 1/4]: cispi(phase) = cispi(r/s) where r/s = phase
        return (phase_num, phase_den, Q1)

    elseif 2 * phase_num <= phase_den
        # phase ∈ (1/4, 1/2]: cispi(phase) = cispi(1/2 - r/s)
        # = i * cispi(-r/s) where r/s = 1/2 - phase
        # r/s = (phase_den - 2*phase_num) / (2*phase_den)
        base_num = phase_den - 2 * phase_num
        base_den = 2 * phase_den
        g3 = gcd(base_num, base_den)
        return (base_num ÷ g3, base_den ÷ g3, ImQ4)

    elseif 4 * phase_num <= 3 * phase_den
        # phase ∈ (1/2, 3/4]: cispi(phase) = cispi(1/2 + r/s)
        # = i * cispi(r/s) where r/s = phase - 1/2
        # r/s = (2*phase_num - phase_den) / (2*phase_den)
        base_num = 2 * phase_num - phase_den
        base_den = 2 * phase_den
        g3 = gcd(base_num, base_den)
        return (base_num ÷ g3, base_den ÷ g3, ImQ1)

    elseif phase_num <= phase_den
        # phase ∈ (3/4, 1]: cispi(phase) = cispi(1 - r/s)
        # = -cispi(-r/s) where r/s = 1 - phase
        # r/s = (phase_den - phase_num) / phase_den
        base_num = phase_den - phase_num
        g3 = gcd(base_num, phase_den)
        return (base_num ÷ g3, phase_den ÷ g3, NegQ4)

    elseif 4 * phase_num <= 5 * phase_den
        # phase ∈ (1, 5/4]: cispi(phase) = cispi(1 + r/s)
        # = -cispi(r/s) where r/s = phase - 1
        # r/s = (phase_num - phase_den) / phase_den
        base_num = phase_num - phase_den
        g3 = gcd(base_num, phase_den)
        return (base_num ÷ g3, phase_den ÷ g3, NegQ1)

    elseif 2 * phase_num <= 3 * phase_den
        # phase ∈ (5/4, 3/2]: cispi(phase) = cispi(3/2 - r/s)
        # = -i * cispi(-r/s) where r/s = 3/2 - phase
        # r/s = (3*phase_den - 2*phase_num) / (2*phase_den)
        base_num = 3 * phase_den - 2 * phase_num
        base_den = 2 * phase_den
        g3 = gcd(base_num, base_den)
        return (base_num ÷ g3, base_den ÷ g3, NegImQ4)

    elseif 4 * phase_num <= 7 * phase_den
        # phase ∈ (3/2, 7/4]: cispi(phase) = cispi(3/2 + r/s)
        # = -i * cispi(r/s) where r/s = phase - 3/2
        # r/s = (2*phase_num - 3*phase_den) / (2*phase_den)
        base_num = 2 * phase_num - 3 * phase_den
        base_den = 2 * phase_den
        g3 = gcd(base_num, base_den)
        return (base_num ÷ g3, base_den ÷ g3, NegImQ1)

    else
        # phase ∈ (7/4, 2): cispi(phase) = cispi(2 - r/s)
        # = cispi(-r/s) where r/s = 2 - phase
        # r/s = (2*phase_den - phase_num) / phase_den
        base_num = 2 * phase_den - phase_num
        g3 = gcd(base_num, phase_den)
        return (base_num ÷ g3, phase_den ÷ g3, Q4)
    end
end

# Unified twiddle factor expression generator using analytical mathematics
# w_k = e^(-2πik/n) = cispi(-2k/n) for k in ks array
# Returns symbolic representations without computing actual trigonometric values
@inline function get_twiddle_expression(ks::AbstractVector{<:Integer}, n::Integer; T::Type = Float64, accuracy=nothing)
    result = Union{String, Twiddle}[]
    sizehint!(result, length(ks))

    for k in ks
        # Normalize k to 0 <= k < n
        k_norm = mod(k, n)

        # Fast paths for common special cases
        if k_norm == 0
            # cispi(0) = 1
            push!(result, "1")
        elseif 2 * k_norm == n  # k = n/2
            # cispi(-1) = -1
            push!(result, "-1")
        elseif 4 * k_norm == n  # k = n/4
            # cispi(-1/2) = -i
            push!(result, "-im")
        elseif 4 * k_norm == 3 * n  # k = 3n/4
            # cispi(-3/2) = cispi(1/2) = i
            push!(result, "im")
        elseif 8 * k_norm == n  # k = n/8
            # cispi(-1/4) = (1-i)/√2
            push!(result, "INV_SQRT2_Q4")
        elseif 8 * k_norm == 3 * n  # k = 3n/8
            # cispi(-3/4) = -(1+i)/√2
            push!(result, "-INV_SQRT2_Q1")
        elseif 8 * k_norm == 5 * n  # k = 5n/8
            # cispi(-5/4) = cispi(3/4) = -(1-i)/√2
            push!(result, "-INV_SQRT2_Q4")
        elseif 8 * k_norm == 7 * n  # k = 7n/8
            # cispi(-7/4) = cispi(1/4) = (1+i)/√2
            push!(result, "INV_SQRT2_Q1")
        else
            # General case: classify analytically
            push!(result, classify_twiddle_factor(k_norm, n))
        end
    end

    return result
end


@inline function extract_plan_data(plan::T) where T
    if !(:n in fieldnames(T)) || !(:operations in fieldnames(T))
        error("Invalid plan type: missing required fields")
    end
    return (n=plan.n, operations=plan.operations)
end

end

# COMMENTS: IN ORDER TO HAVE NO HEAP USAGE THE PLANNER MUST CREATE THE TESTING MODULE AND NOT A POSSIBLE DYNAMIC MODULE TO BE TESTING UPON POTENTIAL PLANS!!!!
# FOR STATIC ARRAYS OR NOT


