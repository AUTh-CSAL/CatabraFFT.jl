module RadixGenerator

include("helper_tools.jl")
include("radix_plan.jl")
include("fft_seed.jl")
#include("radix_exec.jl")

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
            #cp = cospi(num/den)
            #sp = sinpi(num/den)
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
            end
        end
        current_n >>= 1
    end
    
    # Fallback to numerical representation with high precision if everything else fails
    return "($(round(real_part, digits=16))$(sign_str(imag_part))$(abs(round(imag_part, digits=16)))*im)"
end

@inline function create_D_kernel_square(n::Int, ::Type{T}) where T <: AbstractFloat
    # Pre-allocate the array for twiddle factors
    w = cispi.(T(-2/(n*n)) * collect(1:n-1))
    d = zeros(Complex{T}, (n-1)*(n-1))
    
    # Fill the first row
    @inbounds d[1:n-1] .= w

    # Fill subsequent rows
    @inbounds @simd for j in 2:n-1
        row_start = (j-1)*(n-1)
        prev_row_start = (j-2)*(n-1)
        @views d[row_start+1:row_start+n-1] .= w .* d[prev_row_start+1:prev_row_start+n-1]
    end

    # Reshape the array into a matrix
    d_matrix = reshape(d, (n-1, n-1))

    # Create a collapsed array containing only the upper triangular elements
    num_elements = div((n-1) * n, 2)
    collapsed_array = Vector{Complex{T}}(undef, num_elements)

    # Fill the collapsed array with upper triangular elements
    index = 1
    @inbounds for i in 1:n-1
        @inbounds for j in i:n-1
            collapsed_array[index] = d_matrix[i, j]
            index += 1
        end
    end

    # Generate array of strings from collapsed array
    element_strings = String[]
    @inbounds for i in 1:length(collapsed_array)
        element_expr = get_constant_expression(collapsed_array[i], n*n)
        # Clean up parameter expressions
        element_expr = replace(string(element_expr), r"Expr\(:parameters,.*?\)" => "")
        push!(element_strings, element_expr)
    end

    return element_strings
end


@inline function create_D_kernel_non_square(n1::Int, n2::Int, ::Type{T}) where T <: AbstractFloat
    # Pre-allocate the array for twiddle factors
    m, p = min(n1, n2), max(n1, n2)
    w = cispi.(T(-2/(p*m)) * collect(1:m-1))
    d = zeros(Complex{T}, (p-1)*(m-1))

    # Fill the first row
    @inbounds d[1:m-1] .= w

    # Fill subsequent rows
    @inbounds @simd for j in 2:p-1
        row_start = (j-1)*(m-1)
        prev_row_start = (j-2)*(m-1)
        @views d[row_start+1:row_start+m-1] .= w .* d[prev_row_start+1:prev_row_start+m-1]
    end

    # Reshape the array into a matrix
    d_matrix = reshape(d, (m-1, p-1))

    # Generate an array of strings from the matrix elements
    element_strings = String[]

    if n2 == 2
        # Vector case
        @inbounds for i in 1:size(d_matrix, 1)
            @inbounds for j in 1:size(d_matrix, 2)
                element_expr = get_constant_expression(d_matrix[i, j], n1*n2)
                element_expr = replace(string(element_expr), r"Expr\(:parameters,.*?\)" => "")
                push!(element_strings, element_expr)
            end
        end
    else
        # Matrix case
        @inbounds for i in 1:size(d_matrix, 1)
            row_elements = String[]
            @inbounds for j in 1:size(d_matrix, 2)
                element_expr = get_constant_expression(d_matrix[i, j], n1*n2)
                element_expr = replace(string(element_expr), r"Expr\(:parameters,.*?\)" => "")
                push!(row_elements, element_expr)
            end
            # Add the entire row as a single string with elements separated by spaces
            push!(element_strings, join(row_elements, " "))
        end
    end

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
function generate_kernel_names(radix::Int, suffixes::Vector{String}, stride::Int)
    if suffixes == ["mat"]
        base = "fft$(radix)_$(stride)!"
        #suffix = join(suffixes, "_")
        #return (string(base, isempty(suffix) ? "" : "_$suffix", "!"), string(base, "!"))
        return base
    else
        base = "fft$(radix)_shell"
        suffix = join(suffixes, "_")
        return (string(base, isempty(suffix) ? "" : "_$suffix", "!"), string(base, "!"))
    end
end


# Function to generate function signature
function generate_signature(suffixes::Vector{String}, ::Type{T}) where T <: AbstractFloat
    y_only = "y" in suffixes
    layered = "layered" in suffixes
    
    if y_only
        return "(y::AbstractVector{Complex{$T}})"
    elseif layered
        return "(y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}}, s::Int, n1::Int, theta::$T=$T(0.125))"
    else
        return "(y::AbstractArray{Complex{$T}, 1}, x::AbstractArray{Complex{$T}, 1})"
    end
end

# Main function to generate kernel code
function generate_kernel(radix::Int, op, suffixes::Vector{String}, stride::Int, ::Type{T}) where T <: AbstractFloat
    @show suffixes
    if "mat" ∈ suffixes
        println("WORKS")
        @show name = generate_kernel_names(radix, suffixes, stride)
        signature = generate_signature(suffixes, T)
        @show D =  generate_D_kernel(op, T)
        kernel_code = makefftradix(radix, suffixes, D, T)
        return """
        @inline function $(name)$signature 
            @inbounds  begin
            $kernel_code
            end
        end
        """
    else
    @show names = generate_kernel_names(radix, suffixes, 0)

    signature = generate_signature(suffixes, T)
    @show signature

    kernel_code = makefftradix(radix, suffixes, String[], T)
    
    # Generate the complete linear function
    return """
    @inline function $(names[2])$signature 
        @inbounds begin
        $kernel_code
        end
    end
    """
    end
end

# ENCHANT
function generate_all_kernels(plan_data::NamedTuple, ::Type{T}; suffix_combinations::Union{Nothing, Vector{Vector{String}}}=nothing) where T <: AbstractFloat
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

    #=
    if isnothing(suffix_combinations)
        suffix_combinations = Vector{Vector{String}}([
            String[],
            ["mat"],
            ["layered"]
        ])
    end
    =#

    for (i, (rad, op)) in enumerate(zip(radices, plan_data.operations))
        println("Index: $i, Radix: $rad, Operation: $op")
    end
    
    @show radices, symbols, suffix_combinations, plan_data.operations
    kernels = Vector{String}()
    if suffix_combinations == [String[]] #linear order
        push!(kernels, generate_kernel(radices[1], plan_data.operations[1], suffix_combinations[1], 0, T))
    elseif suffix_combinations == [["mat"]]
        for (i, (rad, op)) in enumerate(zip(radices, plan_data.operations))
            stride = op.n_groups ÷ rad
            for s in 1:stride
                @show s
                push!(kernels, generate_kernel(rad, op, suffix_combinations[1], s, T))
            end
        end
    end
    
    return kernels
end

# MEASURE
#=
function generate_all_kernels(N::Int,  ::Type{T}; suffix_combinations::Union{Nothing, Vector{Vector{String}}}=nothing) where T <: AbstractFloat
    if N < 2 || (N & (N - 1)) != 0  # Check if N is less than 2 or not a power of 2
        error("N must be a power of 2 and greater than or equal to 2")
    end
        
    radices = subpowers_of_two(N)

    if isnothing(suffix_combinations)
    suffix_combinations = Vector{Vector{String}}([
        #String[],
        ["mat"],
        #["ivdep"],
        #["y"],
        #["y", "ivdep"],
        #["layered"], # Must have generated normal kernels to produces functional layered kernels
        #["layered", "ivdep"]
    ])
    end
    
    kernels = Vector{String}()
    
    @inbounds for radix ∈ radices
        @inbounds for suffixes ∈ suffix_combinations
            push!(kernels, generate_kernel(radix, suffixes, T))
        end
    end
    
    return kernels
end
=#

function generate_D_kernel(op, ::Type{T}) where T <: AbstractFloat
    s = op.stride
    if s == 1
        return String[]
    else
        n_group = op.n_groups
        return n_group == s ? create_D_kernel_square(s, T) : create_D_kernel_non_square(s, n_group, T)
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
    module_constants = generate_module_constants(plan_data.n, T)
    custom_combinations = length(plan_data.operations) == 1 ? [String[]] : [["mat"]]
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
