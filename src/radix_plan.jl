module Radix_Plan

using LoopVectorization, AbstractFFTs

export RadixPlan

# Struct to hold a single FFT operation
struct FFTOp
    op_type::Symbol  # :fft16, :fft8, :fft4, :fft3,  :fft2
    input_buffer::Symbol  # :x or :y
    output_buffer::Symbol # :x or :y
    stride::Int
    n_groups::Int
    eo::Bool
end

# Struct to hold the complete FFT plan
struct RadixPlan{T<:AbstractFloat} <: AbstractFFTs.Plan{T}
    operations::Vector{FFTOp}
    n::Int
end

function create_radix_2_plan(n::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    use_radix16 = n < 2^15 ? true : false


    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1
        if use_radix16
            if remaining_n % 16 == 0 && remaining_n ≥ 16
                radix = 16
            end
        end
        # Choose largest possible radix (8, 4, or 2)
        if remaining_n % 8 == 0 && remaining_n ≥ 8
            radix = 8
        elseif remaining_n % 4 == 0 && remaining_n ≥ 4
            radix = 4
        else
            radix = 2
        end

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

function create_radix_3_plan(n::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1
        if remaining_n % 9 == 0 && remaining_n ≥ 9
            radix = 9
        else
            radix = 3
        end

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

# Simple universal planner for basic modulo radix planning
function create_radix_plan(n::Int, radix::Int, ::Type{T}) where T <: AbstractFloat
    operations = FFTOp[]

    # Determine the sequence of radix operations
    remaining_n = n
    current_stride = 1
    input_buffer = :x
    output_buffer = :y
    eo = false

    while remaining_n > 1

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
