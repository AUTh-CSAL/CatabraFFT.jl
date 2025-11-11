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
                    @show D, typeof(D)
                    name, expr = generate_kernel_expression(radix, op, suffix_combinations, p, D, false, T) # Generate appropriate sub-kernel by embedding D matrix layer on it at compile-time!
                    kernels[name] = expr
                end
            else
                vec_suffix = add_flag(suffix_combinations, VEC)
                for p in 0:(n_kernels_needed-1)
                    name, expr = generate_kernel_expression(radix, op, vec_suffix, p, Vector{T}, true, T)
                    kernels[name] = expr
                end
            end
        end
    end
    
    return kernels
end

@inline function generate_kernel_expression(radix::Int, op, suffixes::SuffixFlags, p::Int, D::Union{Vector{T}, Type{<:AbstractVector}}, is_last::Bool, ::Type{T}) where T <: AbstractFloat
    if op.eo && is_last
        suffixes = add_flag(suffixes, Y)
    end
    
    name = generate_kernel_name(radix, suffixes, p, op)
    SIZE = op.n_groups * op.stride
    SIMD_BITS = 256
    @show D, typeof(D)
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

function generate_D_kernel(p, radix::Int, current_stride::Int, next_stride::Int, n_groups::Int, ::Type{T})::Union{Vector{T}, Type{<:AbstractVector}} where T <: AbstractFloat
    (next_stride == 1 || n_groups == 1 || p == 1) && return Vector{T}[]

    D_flat = create_D_kernel(radix, current_stride, next_stride, n_groups, T)
    # D_flat is organized as: for each group j in [0, n_groups-1], all radix twiddles for that group
    # Layout: [(k=0,j=0), (k=1,j=0), ..., (k=radix-1,j=0), (k=0,j=1), (k=1,j=1), ..., (k=radix-1,j=n_groups-1)]
    # Extract column p (1-indexed), which corresponds to group j = p-1
    result = Vector{T}()
    sizehint!(result, 2*(radix-1))

    j = p - 1  # Convert to 0-indexed group
    @inbounds @simd for k in 1:(radix-1)  # Skip k=0 (identity), extract k=1 to radix-1
        # Twiddle for (k, j) is at indices: 2*j*radix + 2*k + 1 (cos), 2*j*radix + 2*k + 2 (sin)
        cos_idx = 2*j*radix + 2*k + 1
        sin_idx = 2*j*radix + 2*k + 2
        push!(result, D_flat[cos_idx])    # cos
        push!(result, D_flat[sin_idx])    # sin
    end

    return result
end

@inline function create_D_kernel(radix::Int, current_stride::Int, next_stride::Int, n_groups::Int, ::Type{T})::Vector{T} where T <: AbstractFloat
    N = next_stride * n_groups
    n_elements = radix * n_groups

    # Output: [cos1, sin1, cos2, sin2, ...] interleaved
    d_real = Vector{T}(undef, 2 * n_elements)

    if USE_IVM && n_elements > 16
        # Use IntelVectorMath for vectorized cos/sin computation
        phases = Vector{T}(undef, n_elements)
        phase_factor = T(-2 / N)

        idx = 1
        @inbounds @simd for j in 0:(n_groups-1)
            for k in 0:(radix-1)
                phase_val = phase_factor * k * current_stride * j
                # Avoid -0.0 which can cause issues
                phases[idx] = iszero(phase_val) ? zero(T) : phase_val
                idx += 1
            end
        end

        # Compute cos and sin using IVM
        cos_vals, sin_vals = fast_cossinpi(phases, T)

        # Interleave cos and sin
        @inbounds for i in 1:n_elements
            d_real[2*i - 1] = cos_vals[i]
            d_real[2*i] = sin_vals[i]
        end
    else
        # Scalar fallback
        phase = T(-2 / N)
        idx = 1
        @inbounds for j in 0:(n_groups-1)
            for k in 0:(radix-1)
                angle = phase * k * current_stride * j
                d_real[idx] = cospi(angle)
                d_real[idx + 1] = sinpi(angle)
                idx += 2
            end
        end
    end

    return d_real
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

#=
function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    #if USE_IVM && n > 16
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

=#

"""
get_twiddle_expression(ks, n; T=Float32, accuracy=nothing)

Compute twiddle factors for indices `ks` (vector of integers) for transform length `n`.
Returns a Vector of tuples (wr, wi) where wr = cos(-2π*k/n), wi = sin(-2π*k/n),
computed using IntelVectorMath.jl (IVM) mutating APIs for best throughput.

Arguments
- ks : Vector{<:Integer} — indices (supports zero-based ks like 0:(n/2-1))
- n  : Int — FFT length (denominator of angle)
- T  : Float32 or Float64 (default Float32) — element type for trig evaluation
- accuracy : optional symbol to set VML accuracy, e.g. :HA (high), :LA (low), :EP (enhanced perf)

Return
- Vector{Tuple{T,T}} where each entry is (cosθ, sinθ) for θ = -2π * k / n
"""
function get_twiddle_expression(ks::AbstractVector{<:Integer}, n::Integer; T::Type = Float32, accuracy=nothing)
    len = length(ks)
    if len == 0
        return Vector{Tuple{T,T}}()
    end

    # Prepare angle array (θ = -2π * k / n) as T
    angles = Vector{T}(undef, len)
    two_pi = T(2pi)
    # ks may be zero-based (you used collect(0:n2-1)); preserve that semantics
    @inbounds for i in 1:len
        k = ks[i]
        angles[i] = -two_pi * T(k) / T(n)
    end

    # Optionally control accuracy/mode (wrap IVM calls; recommended values: :HA, :LA, :EP)
    # Use IVM.vml_set_accuracy if caller wants to tune speed vs accuracy.
    # Map friendly symbols to IVM constants if provided
    if !isnothing(accuracy)
        # allowed symbols: :HA, :LA, :EP  (matches Intel VML accuracy modes)
        try
            if accuracy === :LA
                IVM.vml_set_accuracy(IVM.VML_LA)
            elseif accuracy === :EP
                IVM.vml_set_accuracy(IVM.VML_EP)
            elseif accuracy === :HA
                IVM.vml_set_accuracy(IVM.VML_HA)
            else
                @warn "Unknown accuracy symbol; ignoring" accuracy
            end
        catch e
            @warn "Could not set IVM accuracy: $e"
        end
    end

    # Allocate destination buffers (mutating, no extra allocations other than these)
    cosbuf = Vector{T}(undef, len)
    sinbuf = Vector{T}(undef, len)

    # Compute cos and sin via IntelVectorMath in-place functions (fast, threaded)
    # The mutating (!) forms accept 1D strided arrays and are much faster than broadcasting.
    # Example: IVM.cos!(cosbuf, angles); IVM.sin!(sinbuf, angles)
    IVM.cos!(cosbuf, angles)
    IVM.sin!(sinbuf, angles)

    return [get_constant_expression(Complex{T}(cosbuf[i], sinbuf[i]), n) for i in 1:len]
end

end

# COMMENTS: IN ORDER TO HAVE NO HEAP USAGE THE PLANNER MUST CREATE THE TESTING MODULE AND NOT A POSSIBLE DYNAMIC MODULE TO BE TESTING UPON POTENTIAL PLANS!!!!
# FOR STATIC ARRAYS OR NOT


