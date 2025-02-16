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
    # Handle single-digit inputs directly
    n ≤ 9 && return [create_radix_plan_from_decomposition(n, [n], T)]

    if n > maximum(valid_radices)
        @assert all(n % radix == 0 for radix in valid_radices) "n must be divisible by all radices"
    end

    decompositions = Vector{Vector{Int}}()
    
    function backtrack(remaining::Int, current::Vector{Int}, last_radix::Int)
        remaining == 1 && push!(decompositions, copy(current))
        for radix in valid_radices
            radix ≤ last_radix && remaining % radix == 0 && begin
                push!(current, radix)
                backtrack(remaining ÷ radix, current, radix)
                pop!(current)
            end
        end
    end
    
    backtrack(n, Int[], typemax(Int))

    # Get minimul element of valid radices list
    min_elem = minimum(valid_radices)

    # Strict filtering logic
    function is_valid(decomp)
        # Allow single-element decompositions
        length(decomp) == 1 && return true

        # Reject any decomposition starting with min element
        decomp[1] == min_elem && return false

        # Reject non-uniform 4-based decompositions
        decomp[1] == 4 && decomp != fill(4, length(decomp)) && return false

        # Universal min elem ratio check
        count_mins = count(==(min_elem), decomp)
        max_allowed_mins = length(decomp) ÷ 2  # Integer division
        count_mins > max_allowed_mins && return false

        true
    end

    filtered = filter(is_valid, decompositions)
    filtered = [[n]]
    @show filtered
    
    # Create RadixPlan objects
    [create_radix_plan_from_decomposition(n, decomp, T) for decomp in filtered]
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
