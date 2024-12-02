module Radix_Execute

using Core.Compiler: Core, return_type
using RuntimeGeneratedFunctions
using ..Radix_Plan

RuntimeGeneratedFunctions.init(Radix_Execute)

include("radix_2_codelets.jl")
include("radix_3_codelets.jl")
include("radix_5_codelets.jl")
include("radix_7_codelets.jl")

export generate_execute_function!

using LoopVectorization

function generate_safe_execute_function!(plan::RadixPlan)
    current_input = :x
    current_output = :y
    ops = []

    for (i, op) in enumerate(plan.operations)
        if op.op_type == :fft9
                if op === last(plan.operations)
                    if op.eo
                        push!(ops,  :(radix3_family.fft9_shell_y!($(current_input), $(op.stride))))
                    else
                        push!(ops,  :(radix3_family.fft9_shell!($(current_output), $(current_input), $(op.stride))))
                    end
                else
                    n1 = op.n_groups ÷ 9
                    theta = 2 / op.n_groups
                    push!(ops, :(radix3_family.fft9_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
                end
            elseif op.op_type == :fft8
                if op === last(plan.operations)
                if op.eo
                    push!(ops,  :(radix2_family.fft8_shell_y!($(current_input), $(op.stride))))
                else
                    push!(ops,  :(radix2_family.fft8_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                    n1 = op.n_groups >> 3
                    theta = 2 / op.n_groups
                    if theta == 0.125
                        push!(ops, :(radix2_family.fft8_shell_layered_theta_1_8!($(current_output), $(current_input), $(op.stride), $n1)))
                    else
                        push!(ops, :(radix2_family.fft8_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
                    end
            end
            elseif op.op_type == :fft7
                if op === last(plan.operations)
                if op.eo
                    push!(ops, :(radix7_family.fft7_shell_y!($(current_input), $(op.stride))))
                else
                    push!(ops, :(radix7_family.fft7_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                    n1 = op.n_groups ÷ 7
                    theta = 2 / op.n_groups
                    push!(ops, :(radix7_family.fft7_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
            end
            elseif op.op_type == :fft5
                if op === last(plan.operations)
                    if op.eo
                    push!(ops, :(radix5_family.fft5_shell_y!($(current_input), $(op.stride))))
                else
                    push!(ops, :(radix5_family.fft5_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                    n1 = op.n_groups ÷ 5
                    theta = 2 / op.n_groups
                    push!(ops,  :(radix5_family.fft5_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
            end
            elseif op.op_type == :fft4
                if op.eo
                    push!(ops, :(radix2_family.fft4_shell_y!($(current_input), $(op.stride))))
                else
                    push!(ops, :(radix2_family.fft4_shell!($(current_output), $(current_input), $(op.stride))))
                end
            elseif op.op_type == :fft3
                    if op.eo
                    push!(ops, :(radix3_family.fft3_shell_y!($(current_input), $(op.stride))))
                else
                    push!(ops, :(radix3_family.fft3_shell!($(current_output), $(current_input), $(op.stride))))
                end
            elseif op.op_type == :fft2
                if op.eo
                push!(ops, :(radix2_family.fft2_shell_y!($(current_input), $(op.stride))))
            else
                push!(ops,  :(radix2_family.fft2_shell!($(current_output), $(current_input), $(op.stride))))
            end
        end
        current_input, current_output = current_output, current_input
    end


    function_body = Expr(:block, ops...)

    # Combine all operations into a single function
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)

    #@show ex

    runtime_generated_function = @RuntimeGeneratedFunction(ex)

    return runtime_generated_function
end

end
