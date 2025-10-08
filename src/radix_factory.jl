
#COMMENTS: IN ORDER TO HAVE NO HEAP USAGE THE PLANNER MUST CREATE THE TESTING MODULE AND NOT A POSSIBLE DYNAMIC MODULE TO BE TESTING UPON POTENTIAL PLANS!!!!
# FOR STATIC ARRAYS OR NOT
####

# Modified RadixGenerator to work without runtime module generation
module RadixGenerator

include("helper_tools.jl")
include("radix_plan.jl")
include("suffix.jl")
include("fft_seed.jl")

using LoopVectorization, SIMD
using .Radix_Plan

export create_kernel_dictionary, extract_plan_data

# Instead of generating a module, generate a dictionary of kernel expressions
function create_kernel_dictionary(plan_data::NamedTuple, ::Type{T})::Dict{String, Expr} where T <: AbstractFloat
    kernels = Dict{String, Expr}()
    
    # Generate constants as local variables
    #constants = generate_local_constants(plan_data.n, T)
    
    # Generate kernel expressions
    custom_combinations = empty_flags()
    if length(plan_data.operations) != 1 
        custom_combinations = add_flag(custom_combinations, MAT) 
    end
    
    kernel_codes = generate_all_kernel_expressions(plan_data, T; suffix_combinations=custom_combinations)
    
    # Combine constants with each kernel
    for (name, code) in kernel_codes
        #kernels[name] = Expr(:block, constants..., code)
        kernels[name] = Expr(:block, code)
    end
    
    return kernels
end

# Generate constants dictionary for compile-time embedding
"""
 - Compile-Time Constant Embedding: The $(:COSPI_1_8) syntax embeds the actual literal value (like 0.9238795f0) directly into the generated expression at compile time.

 - Zero Runtime Overhead: No variable lookups, no stack allocation for constants - just immediate values in the assembly code.

 - Only Used Constants: The dictionary approach means you only generate constants that might actually be needed for size N, avoiding waste.

 - Type-Safe: Constants are generated with the correct type (Float32 vs Float64) at compile time.
"""
function generate_local_constants_dict(n::Int, ::Type{T}) where T <: AbstractFloat
    @assert ispow2(n) "n must be a power of 2"
    constants_dict = Dict{Symbol, T}()
    
    # Collect all unique fractions that could appear
    fractions = Set{Tuple{Int,Int}}()
    
    # Add fractions from basic twiddle factors W_n^k = cispi(-k/n*2)
    for k in 0:(n÷2-1)
        if k == 0 continue end  # Skip k=0
        
        # Reduce fraction k/(n/2) to lowest terms
        gcd_val = gcd(k, n÷2)
        num = k ÷ gcd_val
        den = (n÷2) ÷ gcd_val
        
        push!(fractions, (num, den))
    end
    
    # Add fractions from Stockham algorithm's intermediate stages
    current_n = n
    while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        
        for i in 1:2:s
            num = n4 - i
            den = n2
            
            # Reduce to lowest terms
            gcd_val = gcd(abs(num), den)
            reduced_num = abs(num) ÷ gcd_val
            reduced_den = den ÷ gcd_val
            
            push!(fractions, (reduced_num, reduced_den))
        end
        current_n >>= 1
    end
    
    # Generate constants for all collected fractions
    for (num, den) in fractions
        angle_cos = T(cospi(num/den))
        angle_sin = T(sinpi(num/den))
        
        cospi_name = Symbol("COSPI_$(num)_$(den)")
        sinpi_name = Symbol("SINPI_$(num)_$(den)")
        
        constants_dict[cospi_name] = angle_cos
        constants_dict[sinpi_name] = angle_sin
    end
    
    if n >= 8
        constants_dict[:INV_SQRT2] = T(1/sqrt(2))
    end
    
    return constants_dict
end

# Modified to return expressions instead of string code
function generate_all_kernel_expressions(plan_data::NamedTuple, ::Type{T}; 
                                        suffix_combinations::Union{Nothing, SuffixFlags}=nothing) where T <: AbstractFloat
    kernels = Dict{String, Expr}()
    
    if has_flag(suffix_combinations, NONE)
        op = plan_data.operations[1]
        radix = get_radix_divisor(op.op_type)
        name, expr = generate_kernel_expression(radix, op, suffix_combinations, 0, String[], true, T)
        kernels[name] = expr
        
    elseif has_flag(suffix_combinations, MAT)
        for (stage_idx, op) in enumerate(plan_data.operations)
            radix = get_radix_divisor(op.op_type)
            is_final = (stage_idx == length(plan_data.operations))
            
            # Calculate SIZE for this stage
            SIZE = op.n_groups * op.stride
            n_kernels_needed = SIZE ÷ radix
            
            if !is_final
                # Non-final stage: generate with D matrix twiddles
                next_op = plan_data.operations[stage_idx + 1]
                n_unique_d_columns = next_op.n_groups
                
                for p in 0:(n_kernels_needed-1)
                    # Cycle through D matrix columns
                    d_column_idx = (p % n_unique_d_columns) + 1
                    D = generate_D_kernel(d_column_idx, next_op.stride, next_op.n_groups, T)
                    @show D, op
                    name, expr = generate_kernel_expression(radix, op, suffix_combinations, p, D, false, T)
                    kernels[name] = expr
                end
            else
                # Final stage: VEC version
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

# Modified to return expression instead of string
function generate_kernel_expression(radix::Int, op, suffixes::SuffixFlags, p::Int, D, is_last::Bool, ::Type{T}) where T <: AbstractFloat
    if op.eo && is_last
        suffixes = add_flag(suffixes, Y)
    end
    
    name = generate_kernel_name(radix, suffixes, p, op)  # Use singular function
    
    SIZE = op.n_groups * op.stride
    SIMD_BITS = 256
    kernel_body = makefftradix(radix, suffixes, D, p, op, SIZE, T, SIMD_BITS)
    #kernel_body = makefftradix_simd(radix, suffixes, D, p, op.stride, SIZE, T)
    #@show kernel_body
    
    return name, kernel_body
end

# Fixed function to always return a single string
function generate_kernel_name(radix::Int, suffix_flags::SuffixFlags, p::Int, op)
    has_mat = has_flag(suffix_flags, MAT)
    has_y = has_flag(suffix_flags, Y)
    has_vec = has_flag(suffix_flags, VEC)
    has_layered = has_flag(suffix_flags, LAYERED)
    
    s = op.stride
    n_g = op.n_groups
    
    if has_mat && !has_layered 
        return "fft$(radix)_$(s)x$(n_g)_$(p)!"
    end
    
    base = "fft$(radix)_shell"
    
    if is_empty(suffix_flags)
        kernel_name = base
    else
        active_flags = get_active_flags(suffix_flags)
        suffix_parts = [flag_to_string(flag) for flag in active_flags if flag != NONE]
        suffix = join(suffix_parts, "_")
        kernel_name = string(base, "_", suffix)
    end
    
    return string(kernel_name, "!")
end

function generate_D_kernel(p, s, n1, ::Type{T}) where T <: AbstractFloat
    if s == 1 || n1 == 1
        return String[]
    else
        if p == 1 
            return String[]
        else
            D_flat = create_D_kernel(s, n1, T)
            D_matrix = reshape(D_flat, s, n1)
            return view(D_matrix, :, p-1)
        end
    end
end

@inline function create_D_kernel(n1::Int, n2::Int, ::Type{T}) where T <: AbstractFloat
    d_matrix = Matrix{Complex{T}}(undef, n1, n2)
    phase = T(-2 / (n1 * n2))
    @inbounds for i in 1:n1
        @inbounds for j in 1:n2
            d_matrix[i, j] = cispi(phase * (i) * (j))
        end
    end

    element_strings = String[]
    @inbounds for elem in d_matrix
        expr = get_constant_expression(elem, n1*n2)
        clean_expr = replace(string(expr), r"Expr\(:parameters,.*?\)" => "")
        push!(element_strings, clean_expr)
    end

    return element_strings
end

function get_constant_expression(w::Complex{T}, n::Integer)::String where T <: AbstractFloat
    real_part = real(w)
    imag_part = imag(w)
    
    isclose(a, b) = (abs(real(a) - real(b)) < eps(T) * 20) && (abs(imag(a) - imag(b)) < eps(T) * 20)
    sign_str(x) = x ≥ 0 ? "+" : "-"
    
    # Check common cases first
    common_cases = [
        (1.0, 0.0) => "1",
        (-1.0, 0.0) => "-1",
        (0.0, 1.0) => "im", 
        (0.0, -1.0) => "-im",
        (1/√2, 1/√2) => "INV_SQRT2_Q1",
        (1/√2, -1/√2) => "INV_SQRT2_Q4",
        (-1/√2, 1/√2) => "-INV_SQRT2_Q4", 
        (-1/√2, -1/√2) => "-INV_SQRT2_Q1"
    ]
    
    for ((re, im), expr) in common_cases
        if isclose(real_part, re) && isclose(imag_part, im)
            return expr
        end
    end
    
    # Math trick: Check all twiddle factors W_n^k = cispi(-2k/n) for k = 0 to n/2-1
    # This covers ALL possible twiddle factors that can appear in FFT of size n
    for k in 0:(n÷2-1)
        if k == 0 continue end  # Already covered by common cases
        
        # Reduce fraction to lowest terms
        gcd_val = gcd(k, n÷2)
        num = k ÷ gcd_val  
        den = (n÷2) ÷ gcd_val
        
        # Basic twiddle: W_n^k = cispi(-2k/n) = cispi(-num/den)
        w_basic = cispi(-num/den)
        
        # Check all four rotations: 1, i, -1, -i multipliers
        if isclose(w, w_basic)
            return "CISPI_$(num)_$(den)_Q4"  # Negative angle → Q4
        elseif isclose(w, -w_basic)
            return "-CISPI_$(num)_$(den)_Q4"
        elseif isclose(w, im * w_basic)
            return "im*CISPI_$(num)_$(den)_Q4"  # This matches your case!
        elseif isclose(w, -im * w_basic)
            return "-im*CISPI_$(num)_$(den)_Q4"
        end
            
        # Also check positive angle version: cispi(num/den)
        w_pos = cispi(num/den)
        if isclose(w, w_pos)
            return "CISPI_$(num)_$(den)_Q1"
        elseif isclose(w, -w_pos)
            return "-CISPI_$(num)_$(den)_Q1" 
        elseif isclose(w, im * w_pos)
            return "im*CISPI_$(num)_$(den)_Q1"
        elseif isclose(w, -im * w_pos)
            return "-im*CISPI_$(num)_$(den)_Q1"
        end
    end
    
    # Fallback to numerical
    return "($(round(real_part, digits=16))$(sign_str(imag_part))$(abs(round(imag_part, digits=16)))*im)"
end

function extract_plan_data(plan::T) where T
    if !(:n in fieldnames(T)) || !(:operations in fieldnames(T))
        error("Invalid plan type: missing required fields")
    end
    return (n=plan.n, operations=plan.operations)
end

# Modified to NOT generate a module but return kernel expressions
function evaluate_fft_generated_module(target_module::Module, plan::P, ::Type{T}) where {P, T <: AbstractFloat}
    # Do nothing - we don't generate modules anymore
    # This function is kept for compatibility but doesn't do anything
    return nothing
end

"""
Generate twiddle factor expressions for a given collection of indices
"""
function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

end