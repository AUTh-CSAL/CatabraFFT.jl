module RadixGenerator

include("helper_tools.jl")
include("radix_plan.jl")
include("suffix.jl")
include("fft_seed.jl")


using LoopVectorization
using .Radix_Plan

export evaluate_fft_generated_module

function generate_module_constants(n::Int, ::Type{T}) where T <: AbstractFloat
    @assert ispow2(n) "n must be a power of 2"
    str = "# Optimized twiddle factors for radix-2^s FFT size $n\n\n"
    current_n = n
    # Only store the minimal set of unique twiddle factors needed
    while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        # Store only unique twiddle factors for this stage
        # We exploit symmetry and periodicity to minimize storage
        str *= "# Stage $current_n constants\n"
        for i in 1:2:s
            # Calculate angle once and reuse
            angle = 2 * (n4-i) / current_n
            angle_cos = T(cospi(angle))
            angle_sin = T(sinpi(angle))
            str *= "const COSPI_$(n4-i)_$(n2)::$T = $angle_cos\n"
            str *= "const SINPI_$(n4-i)_$(n2)::$T = $angle_sin\n"
        end
        str *= "\n"
        current_n >>= 1
    end
    
    # Add only essential special constants
    if n >= 8
        str *= "const INV_SQRT2::$T = $(T(1/sqrt(2)))\n"
    end
    
    return str
end

"""
Enhanced twiddle factor expression generator with improved constant recognition
"""
function get_constant_expression(w::Complex{T}, n::Integer)::String where T <: AbstractFloat
    real_part = real(w)
    imag_part = imag(w)
    
    # Helper for approximate equality
    isclose(a, b) = (abs(real(a) - real(b)) < eps(T) * 10) && (abs(imag(a) - imag(b)) < eps(T) * 10)
    
    # Function to get sign string
    sign_str(x) = x ≥ 0 ? "+" : "-"
    
    # Common cases table with twiddle factors patterns commonly met

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
    
    # Check special cases first
    for ((re, im), expr) in common_cases
        if isclose(real_part, re) && isclose(imag_part, im)
            return expr
        end
    end
    
    current_n = n
    # Handle cases based on radix size
    while current_n >= 16
        n2 = current_n >> 1
        n4 = current_n >> 2
        s = current_n >> 3
        angles = [(n4-i,n2) for i in 1:2:s]
        for (num, den) in angles
            cispi1, cispi2  = cispi(num/den), cispi(-num/den)
            if isclose(w, cispi1)
                return "CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, -cispi1)
                return "-CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, cispi2)
                return "CISPI_$(num)_$(den)_Q4"
            elseif isclose(w, -cispi2)
                return "-CISPI_$(num)_$(den)_Q4"
            elseif isclose(w, -im*cispi1)
                return "-im*CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, -im*cispi2)
                return "-im*CISPI_$(num)_$(den)_Q4"
            elseif isclose(w, im*cispi1)
                return "im*CISPI_$(num)_$(den)_Q1"
            elseif isclose(w, im*cispi2)
                return "im*CISPI_$(num)_$(den)_Q4)"
            end
        end
        current_n >>= 1
    end
    
    # Fallback to numerical representation with high precision if everything else fails
    return "($(round(real_part, digits=16))$(sign_str(imag_part))$(abs(round(imag_part, digits=16)))*im)"
end

@inline function create_D_kernel(n1::Int, n2::Int, ::Type{T}) where T <: AbstractFloat
    # Initialize matrix
    d_matrix = Matrix{Complex{T}}(undef, n1, n2)
    
    # Compute elements directly using i*j/(n1*n2) exponent
    phase = T(-2 / (n1 * n2))
    @inbounds for i in 1:n1
        @inbounds for j in 1:n2
            d_matrix[i, j] = cispi(phase * (i) * (j))
        end
    end
    @show d_matrix

    # Generate constant expressions for all elements
    element_strings = String[]
    @inbounds for elem in d_matrix
        expr = get_constant_expression(elem, n1*n2)
        # CIS CONSTANT NOT DEFINED BECAUSE OF MODULE IMPORT
        clean_expr = replace(string(expr), r"Expr\(:parameters,.*?\)" => "")
        push!(element_strings, clean_expr)
    end
    @show element_strings

    return element_strings
end

"""
Generate twiddle factor expressions for a given collection of indices
"""
function get_twiddle_expression(collect::Vector{Int}, n::Int)::Vector{String}
    wn = cispi.(-2/n * collect)
    return [get_constant_expression(w, n) for w in wn]
end

# Function to generate kernel name
function generate_kernel_names(radix::Int, suffix_flags::SuffixFlags, p::Int, op)
    has_mat = has_flag(suffix_flags, MAT)
    has_y = has_flag(suffix_flags, Y)
    has_vec = has_flag(suffix_flags, VEC)
    has_layared = has_flag(suffix_flags, LAYERED)
    
    s = op.stride
    n_g = op.n_groups
    # Special MAT case
    if has_mat && !has_layared 
        return "fft$(radix)_$(s)x$(n_g)_$(p)!"
    end
    
    # General cases with pattern matching
    base = "fft$(radix)_shell"
    
    if is_empty(suffix_flags)
        kernel_name = base
    else
        active_flags = get_active_flags(suffix_flags)
        suffix_parts = [flag_to_string(flag) for flag in active_flags if flag != NONE]
        suffix = join(suffix_parts, "_")
        kernel_name = string(base, "_", suffix)
    end
    
    return (string(kernel_name, "!"), string(base, "!"))
end

# Function to generate function signature
function generate_signature(suffixes::SuffixFlags, ::Type{T}) where T <: AbstractFloat
    has_y = has_flag(suffixes, Y)
    has_layered = has_flag(suffixes, LAYERED)
    has_vec = has_flag(suffixes, VEC)
    has_mat = has_flag(suffixes, MAT)
    @show has_y has_vec has_mat
    if has_layered
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int, n1::Int, theta::$T=$T(0.125))"
    elseif has_y
        return "(y::AbstractArray{Complex{$T}, 1})"
    else
        return "(y::AbstractArray{Complex{$T}, 1}, x::AbstractArray{Complex{$T}, 1})" 
    end
end

# Main function to generate kernel code
function generate_kernel(radix::Int, op, suffixes::SuffixFlags, p::Int, D, is_last::Bool, ::Type{T}) where T <: AbstractFloat
    if op.eo && is_last
        suffixes = add_flag(suffixes, Y)
    end
    if has_flag(suffixes, NONE)
        name = generate_kernel_names(radix, suffixes, p, op)
        signature = generate_signature(suffixes, T)
        SIZE = op.n_groups * op.stride
        kernel_code = makefftradix(radix, suffixes, D, p, op.stride, SIZE, T) # CREATE EMBEDDED F⊗D KERNEL
        return """
        @inline function $(name[2])$signature 
            @inbounds  begin
            $kernel_code
            end
        end
        """
    else
        name = generate_kernel_names(radix, suffixes, p, op)
        signature = generate_signature(suffixes, T)
        SIZE = op.n_groups * op.stride
        kernel_code = makefftradix(radix, suffixes, D, p, op.stride, SIZE, T)
        # Generate the complete linear function
        return """
        @inline function $name$signature 
            @inbounds begin
            $kernel_code
            end
        end
        """
    end
end

# ENCHANT
function generate_all_kernels(plan_data::NamedTuple, ::Type{T}; suffix_combinations::Union{Nothing, SuffixFlags}=nothing) where T <: AbstractFloat
    # Extract unique radices from the operations in plan_data
    symbols = Vector{Symbol}()
    radices = Vector{Int}()

    @inbounds for op in plan_data.operations
        push!(symbols, op.op_type)  # Add the radix symbol (e.g., :fft64, :fft4)
    end

    # Convert symbols to integers
    @inbounds for symbol in symbols
        # Extract the numeric part of the symbol (e.g., "256" from ":fft256")
        num_str = String(symbol)[4:end]
        push!(radices, parse(Int, num_str))
    end

    kernels = Vector{String}()
    if has_flag(suffix_combinations, NONE) # Linear Order
        push!(kernels, generate_kernel(radices[1], plan_data.operations[1], suffix_combinations, 0, String[], true, T))
    elseif has_flag(suffix_combinations, MAT)
        for (i, (rad, op)) in enumerate(zip(radices, plan_data.operations))
            future_op = i < length(plan_data.operations) ? plan_data.operations[i+1] : nothing
            #=
            if !op.eo
                tmp = op.stride
                op.stride = op.n_groups
                op.n_groups = tmp
                println("SWAPPED")
            end
            =#
            n1 = op.n_groups ÷ rad
            @show rad op n1

            if !isnothing(future_op) 
                op.n_groups, op.stride = future_op.n_groups, future_op.stride # CRITICAL!
                for p in 1:n1
                    D = generate_D_kernel(p, op.stride, op.n_groups, T)
                    push!(kernels, generate_kernel(rad, op, suffix_combinations, p-1, D, false, T))
                end
            else
                # We will add the vectorize suffix_combination for the colunm-wise Fm opeation
                suffix_combinations = add_flag(suffix_combinations, VEC)
                for p in 1:n1
                    push!(kernels, generate_kernel(rad, op, suffix_combinations, p-1, String[], true, T))
                end
            end
        end
    end
    
    return kernels
end

function generate_D_kernel(p, s, n1, ::Type{T}) where T <: AbstractFloat
    if s == 1 || n1 == 1
        return String[]
    else
        if p == 1 
            return String[]
        else
            D_flat = create_D_kernel(s, n1, T)  # 1D vector
            D_matrix = reshape(D_flat, s, n1)   # Reshape to 2D matrix
            return view(D_matrix, :, p-1)       # Now this works
        end
    end
end

# MEASURE KERNEL PRODUCER
function create_kernel_module(N::Int, ::Type{T}) where T <: AbstractFloat
    module_constants = generate_module_constants(N, T)
    custom_combinations = [String[], ["layered"]]
    kernels = generate_all_kernels(N, T; suffix_combinations=custom_combinations)

    family_module_code = """
    module radix_2_family
        Base.@assume_effects :total
        using LoopVectorization

        $module_constants
        
        $(join(kernels, "\n\n"))
    end
    """
    
    return Meta.parse(family_module_code) # Parse directly into an expression
end

# ENCHANT KERNEL PRODUCER
function create_kernel_module(plan_data::NamedTuple, ::Type{T}) where T <: AbstractFloat
    @show plan_data
    module_constants = generate_module_constants(plan_data.n, T)
    custom_combinations = empty_flags()
    if length(plan_data.operations) != 1 custom_combinations = add_flag(custom_combinations, MAT) end
    kernels = generate_all_kernels(plan_data, T; suffix_combinations=custom_combinations)

    family_module_code = """
        module radix_2_family
        using LoopVectorization
        
        $module_constants

        $(join(kernels, "\n\n"))
        end
    """
    
    return Meta.parse(family_module_code)
end


# This function extracts the data we need from any RadixPlan-like type
# by checking for the expected fields
function extract_plan_data(plan::T) where T
    if !(:n in fieldnames(T)) || !(:operations in fieldnames(T))
        error("Invalid plan type: missing required fields")
    end
    return (n=plan.n, operations=plan.operations)
end

# Modify the evaluate_fft_generated_module to use a more flexible type constraint
function evaluate_fft_generated_module(target_module::Module, plan::P, ::Type{T}) where {P, T <: AbstractFloat}
    # Check if the type has the structure we expect
    if !hasfield(P, :n) || !hasfield(P, :operations)
        error("Invalid plan type: must have fields 'n' and 'operations'")
    end
    
    # Create module expression using the extracted data
    module_expr = create_kernel_module(extract_plan_data(plan), T)
    @show module_expr
    Core.eval(target_module, module_expr)
end

function evaluate_fft_generated_module(target_module::Module, n::Int, ::Type{T}) where T <: AbstractFloat
    module_expr = create_kernel_module(n, T)
    Core.eval(target_module, module_expr)
end

end


#COMMENTS: IN ORDER TO HAVE NO HEAP USAGE THE PLANNER MUST CREATE THE TESTING MODULE AND NOT A POSSIBLE DYNAMIC MODULE TO BE TESTING UPON POTENTIAL PLANS!!!!
# FOR STATIC ARRAYS OR NOT
####
