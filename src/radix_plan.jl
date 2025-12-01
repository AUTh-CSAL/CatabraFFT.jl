module Radix_Plan

using LoopVectorization

export RadixPlan

# Struct to hold a single FFT operation using Stockham notation
mutable struct FFTOp
    op_type::Symbol      # :fftN radix body
    input_buffer::Symbol  # :x or :y
    output_buffer::Symbol # :x or :y
    stride::Int          # stride of radix 
    n_groups::Int        # N ÷ stride
    eo::Bool             # even/odd buffer allocation plans
end

# Struct to hold the complete layered FFT plan
struct RadixPlan{T<:AbstractFloat} 
    operations::Vector{FFTOp}
    n::Int
end

function create_all_radix_plans(n::Int, valid_radices::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    if n > maximum(valid_radices)
        @assert all(n % radix == 0 for radix in valid_radices) "n must be divisible by all radices"
    end
    
    decompositions = Vector{Vector{Int}}()
    
    function backtrack(remaining::Int, current::Vector{Int}, last_radix::Int)
        if remaining == 1
            push!(decompositions, copy(current))
            return
        end
        
        @inbounds for radix in valid_radices
            if radix ≤ last_radix && remaining % radix == 0
                push!(current, radix)
                backtrack(remaining ÷ radix, current, radix)
                pop!(current)
            end
        end
    end
    
    backtrack(n, Int[], typemax(Int))
    min_elem = minimum(valid_radices)
    
    function is_valid(decomp)
        # Allow single-element decompositions
        length(decomp) == 1 && return false

        # Check balanced distribution of small radices
        count_mins = count(==(min_elem), decomp)
        max_allowed_mins = length(decomp) ÷ 2 + 1 
        count_mins > max_allowed_mins && return false
        
        return true
    end
    
    filtered = filter(is_valid, decompositions)
    @show filtered
    
    return [create_radix_plan_from_decomposition(n, decomp, T) for decomp in filtered]
end

# Create a RadixPlan from a specific decomposition
function create_radix_plan_from_decomposition(n::Int, decomposition::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    operations = Vector{FFTOp}(undef, length(decomposition))
    
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false
    remaining_n = n
    
    @inbounds for (i, radix) in enumerate(decomposition)
        operations[i] = FFTOp(
            Symbol("fft", radix),
            input_buffer,
            output_buffer,
            current_stride,
            remaining_n,
            eo
        )
        
        # Update state for the next layer
        remaining_n ÷= radix
        current_stride *= radix
        eo = !eo
        input_buffer, output_buffer = output_buffer, input_buffer
    end

    # Obvious fix!
    last_idx = length(decomposition)
    if ((last_idx % 2 == 0) && operations[last_idx].eo) 
        operations[last_idx].output_buffer = :y
    end
    
    return RadixPlan{T}(operations, n)
end

function create_std_radix_plan(n::Int, radices::Vector{Int}, ::Type{T}) where T <: AbstractFloat
    # Sort radices in descending order for greedy selection
    sorted_radices = sort(radices, rev=true)
    
    operations = FFTOp[]
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false
    
    @inbounds while remaining_n > 1
        # Find the largest radix that divides the remaining size
        radix_idx = findfirst(r -> remaining_n % r == 0, sorted_radices)
        radix_idx === nothing && error("Cannot decompose n=$n with the provided radices: $radices")
        
        radix = sorted_radices[radix_idx]
        
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
        remaining_n ÷= radix
        current_stride *= radix
        eo = !eo
        input_buffer, output_buffer = output_buffer, input_buffer
    end
    
    return RadixPlan{T}(operations, n)
end

end
