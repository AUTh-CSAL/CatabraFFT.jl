module Radix_Plan

using LoopVectorization

export RadixPlan

# Struct to hold a single FFT operation using Stockham notation
struct FFTOp
    op_type::Symbol  # :fft16, :fft8, :fft7, :fft5, :fft4, :fft3, :fft2
    input_buffer::Symbol  # :x or :y
    output_buffer::Symbol # :x or :y
    stride::Int
    n_groups::Int
    eo::Bool
end

# Struct to hold the complete layered FFT plan
struct RadixPlan{T<:AbstractFloat} 
    operations::Vector{FFTOp}
    n::Int
end

function create_all_radix_plans(n::Int, valid_radices::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    if n > maximum(valid_radices)
        @assert all(n % radix == 0 for radix in valid_radices) "n must be divisible by all radices in valid_radices"
    end

    decompositions = Vector{Vector{Int}}()
    
    function backtrack(remaining::Int, current::Vector{Int}, last_radix::Int)
        remaining == 1 && push!(decompositions, copy(current))
        for radix in valid_radices
            if radix <= last_radix && remaining % radix == 0 
                push!(current, radix)
                backtrack(remaining ÷ radix, current, radix)
                pop!(current)
            end
        end
    end
    
    backtrack(n, Int[], typemax(Int))

    # Heuristic to narrow valid decomposition solution space to exclude definetly sub-optimal solutions to have to verify
    function is_valid_decomposition(decomposition::Vector{Int})::Bool
    if all(x -> x == 0, mod.(decomposition, 2))
        if length(decomposition) == 1 return true end
        if decomposition[1] == 2 || (decomposition[1] == 4 && decomposition != fill(4, length(decomposition)))
            return false
        end
        count_2s = count(x -> x == 2, decomposition)
        max_2s_allowed = length(decomposition) ÷ 2
        if decomposition[1] in (16, 8) && count_2s > max_2s_allowed
            return false
        end
        true
    end
    end

    filtered_decompositions = filter(is_valid_decomposition, decompositions)
    
    @show filtered_decompositions

    # Create a RadixPlan for each decomposition
    radix_plans = Vector{RadixPlan{T}}()
    for decomposition in filtered_decompositions
        push!(radix_plans, create_radix_plan_from_decomposition(n, decomposition, T))
    end

    radix_plans
end

# Create a RadixPlan from a specific decomposition
function create_radix_plan_from_decomposition(n::Int, decomposition::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    remaining_n = n
    for radix in decomposition
        push!(operations, FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
        ))
        
        # Update state for the next layer
        remaining_n = remaining_n ÷ radix
        current_stride *= radix
        eo = !eo
        input_buffer, output_buffer = output_buffer, input_buffer
    end

    RadixPlan{T}(operations, n)
end

function create_std_radix_plan(n::Int, radices::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Ensure the radices are sorted in descending order
    sorted_radices = sort(radices, rev=true)

    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1
        # Find the largest radix that divides the remaining size
        radix = findfirst(r -> remaining_n % r == 0, sorted_radices)
        if radix === nothing
            error("Cannot decompose n=$n with the provided radices: $radices")
        end
        radix = sorted_radices[radix]

        # Add FFT layer
        push!(operations, FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
        ))

        # Update for next iteration
        remaining_n = remaining_n ÷ radix
        current_stride *= radix
        eo = !eo # swap
        input_buffer, output_buffer = output_buffer, input_buffer
    end

    RadixPlan{T}(operations, n)
end

end
