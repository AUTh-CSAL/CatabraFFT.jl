module RadixGenerator

include("helper_tools.jl")
include("radix_plan.jl")
include("suffix.jl")
include("fft_seed.jl")

using LoopVectorization, SIMD
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
        using IntelVectorMath
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
        return IVM.cis.(radian_phases)
    elseif USE_IVM && T == Float32
        # Convert to Float64, handle -0.0, then convert back
        radian_phases = Float64.(phases) .* π
        @inbounds for i in eachindex(radian_phases)
            if radian_phases[i] == -zero(Float64)
                radian_phases[i] = zero(Float64)
            end
        end
        return ComplexF32.(IVM.cis.(radian_phases))
    else
        return cispi.(phases)
    end
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
    constants_dict = Dict{Symbol, T}()
    
    # Pre-allocate set with estimated size
    fractions = Set{Tuple{Int,Int}}()
    sizehint!(fractions, n >> 1)
    
    # Generate base fractions
    @inbounds for k in 1:(n÷2-1)
        gcd_val = gcd(k, n÷2)
        num = k ÷ gcd_val
        den = (n÷2) ÷ gcd_val
        push!(fractions, (num, den))
    end
    
    # Generate additional fractions for radix-8 patterns
    current_n = n
    @inbounds while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        
        for i in 1:2:s
            num = n4 - i
            den = n2
            gcd_val = gcd(abs(num), den)
            reduced_num = abs(num) ÷ gcd_val
            reduced_den = den ÷ gcd_val
            push!(fractions, (reduced_num, reduced_den))
        end
        current_n >>= 1
    end
    
    # Pre-compute all trig constants
    sizehint!(constants_dict, length(fractions) * 2 + 1)
    @inbounds for (num, den) in fractions
        angle_cos = T(cospi(num/den))
        angle_sin = T(sinpi(num/den))
        constants_dict[Symbol("COSPI_$(num)_$(den)")] = angle_cos
        constants_dict[Symbol("SINPI_$(num)_$(den)")] = angle_sin
    end
    
    if n >= 8
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
        name, expr = generate_kernel_expression(radix, op, suffix_combinations, 0, String[], true, T)
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
                    name, expr = generate_kernel_expression(radix, op, suffix_combinations, p, D, false, T)
                    kernels[name] = expr
                end
            else
                vec_suffix = add_flag(suffix_combinations, VEC)
                for p in 0:(n_kernels_needed-1)
                    name, expr = generate_kernel_expression(radix, op, vec_suffix, p, String[], true, T)
                    kernels[name] = expr
                end
            end
        end
    end
    
    return kernels
end

@inline function generate_kernel_expression(radix::Int, op, suffixes::SuffixFlags, p::Int, D, is_last::Bool, ::Type{T}) where T <: AbstractFloat
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
    if next_stride == 1 || n_groups == 1 || p == 1
        return String[]
    end
    
    D_flat = create_D_kernel(radix, current_stride, next_stride, n_groups, T)
    D_matrix = reshape(D_flat, radix, n_groups)
    return D_matrix[2:radix, p]
end

@inline function create_D_kernel(radix::Int, current_stride::Int, next_stride::Int, n_groups::Int, ::Type{T}) where T <: AbstractFloat
    N = next_stride * n_groups
    d_matrix = Matrix{Complex{T}}(undef, radix, n_groups)
    
    if USE_IVM && radix * n_groups > 16
        phases = Vector{T}(undef, radix * n_groups)
        phase_factor = T(-2 / N)
        
        idx = 1
        @inbounds for j in 0:(n_groups-1)
            for k in 0:(radix-1)
                phase_val = phase_factor * k * current_stride * j
                # *** CRITICAL FIX: Avoid -0.0 which causes IntelVectorMath.cis to fail ***
                # When k=0 or j=0, phase_val becomes -0.0, which must be converted to +0.0
                phases[idx] = iszero(phase_val) ? zero(T) : phase_val
                idx += 1
            end
        end
        
        results = fast_cispi(phases)
        d_matrix[:] = results
    else
        phase = T(-2 / N)
        @inbounds for j in 0:(n_groups-1)
            for k in 0:(radix-1)
                d_matrix[k+1, j+1] = cispi(phase * k * current_stride * j)
            end
        end
    end
    
    # Convert to string expressions
    element_strings = Vector{String}(undef, length(d_matrix))
    @inbounds for i in eachindex(d_matrix)
        element_strings[i] = get_constant_expression(d_matrix[i], N)
    end
    
    return element_strings
end

function get_constant_expression(w::Complex{T}, n::Integer)::String where T <: AbstractFloat
    real_part = real(w)
    imag_part = imag(w)
    tol = eps(T) * 20
    
    @inline isclose(a, b) = abs(a - b) < tol
    
    # Fast path: check simple constants first
    if isclose(real_part, 1.0) && isclose(imag_part, 0.0)
        return "1"
    elseif isclose(real_part, -1.0) && isclose(imag_part, 0.0)
        return "-1"
    elseif isclose(real_part, 0.0) && isclose(imag_part, 1.0)
        return "im"
    elseif isclose(real_part, 0.0) && isclose(imag_part, -1.0)
        return "-im"
    end
    
    # Check sqrt(2) cases
    inv_sqrt2 = T(1/√2)
    if isclose(real_part, inv_sqrt2) && isclose(imag_part, inv_sqrt2)
        return "INV_SQRT2_Q1"
    elseif isclose(real_part, inv_sqrt2) && isclose(imag_part, -inv_sqrt2)
        return "INV_SQRT2_Q4"
    elseif isclose(real_part, -inv_sqrt2) && isclose(imag_part, inv_sqrt2)
        return "-INV_SQRT2_Q4"
    elseif isclose(real_part, -inv_sqrt2) && isclose(imag_part, -inv_sqrt2)
        return "-INV_SQRT2_Q1"
    end
    
    # Check twiddle factors
    n_half = n ÷ 2
    @inbounds for k in 1:(n_half-1)
        gcd_val = gcd(k, n_half)
        num = k ÷ gcd_val  
        den = n_half ÷ gcd_val
        
        w_basic = cispi(T(-num/den))
        re_basic = real(w_basic)
        im_basic = imag(w_basic)
        
        # Check all phase/sign combinations
        if isclose(real_part, re_basic) && isclose(imag_part, im_basic)
            return "CISPI_$(num)_$(den)_Q4"
        elseif isclose(real_part, -re_basic) && isclose(imag_part, -im_basic)
            return "-CISPI_$(num)_$(den)_Q4"
        elseif isclose(real_part, -im_basic) && isclose(imag_part, re_basic)
            return "im*CISPI_$(num)_$(den)_Q4"
        elseif isclose(real_part, im_basic) && isclose(imag_part, -re_basic)
            return "-im*CISPI_$(num)_$(den)_Q4"
        end
        
        w_pos = cispi(T(num/den))
        re_pos = real(w_pos)
        im_pos = imag(w_pos)
        
        if isclose(real_part, re_pos) && isclose(imag_part, im_pos)
            return "CISPI_$(num)_$(den)_Q1"
        elseif isclose(real_part, -re_pos) && isclose(imag_part, -im_pos)
            return "-CISPI_$(num)_$(den)_Q1"
        elseif isclose(real_part, -im_pos) && isclose(imag_part, re_pos)
            return "im*CISPI_$(num)_$(den)_Q1"
        elseif isclose(real_part, im_pos) && isclose(imag_part, -re_pos)
            return "-im*CISPI_$(num)_$(den)_Q1"
        end
    end
    
    # Fallback to literal value
    sign = imag_part >= 0 ? "+" : ""
    return "($(round(real_part, digits=16))$sign$(round(imag_part, digits=16))*im)"
end

@inline function extract_plan_data(plan::T) where T
    if !(:n in fieldnames(T)) || !(:operations in fieldnames(T))
        error("Invalid plan type: missing required fields")
    end
    return (n=plan.n, operations=plan.operations)
end

function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

end

# COMMENTS: IN ORDER TO HAVE NO HEAP USAGE THE PLANNER MUST CREATE THE TESTING MODULE AND NOT A POSSIBLE DYNAMIC MODULE TO BE TESTING UPON POTENTIAL PLANS!!!!
# FOR STATIC ARRAYS OR NOT


