module Radix_Execute

using Core.Compiler: Core, return_type
using RuntimeGeneratedFunctions
using ..Radix_Plan
using Statistics

RuntimeGeneratedFunctions.init(Radix_Execute)

include("radix_2_codelets.jl")
include("radix_3_codelets.jl")
include("radix_5_codelets.jl")
include("radix_7_codelets.jl")


function get_function_reference(op_type::Symbol, variant::Symbol)
    # Map of radix families and their corresponding modules
    radix_modules = Dict(
        :radix2_family => radix2_family,
        :radix3_family => radix3_family,
        :radix5_family => radix5_family,
        :radix7_family => radix7_family
    )
    
    # Construct function name based on operation type and variant
    function_name = Symbol(String(op_type),  "_" * String(variant))
    
    @show function_name
    
    # Try to get the function from the corresponding module
    try
        if hasmethod(getfield(module_ref, function_name), 
                    (AbstractVector{Complex{T}} where T <: AbstractFloat,))
            return getfield(module_ref, function_name)
        end
        error("No matching function found for $function_name")
    catch e
        error("Error accessing function: $function_name. Details: $e")
    end
end

#=
function generate_safe_execute_function!(plan::RadixPlan, show_function=true, check_ivdep=false)
    current_input = :x
    current_output = :y
    ops = []
    ivdep = false

    # A helper function to push the correct operation
    #=
    function push_operation!(ops, op_type, op, current_input, current_output, ivdep)
        radix_family = get_radix_family(op_type)
        @show radix_family
        if op === last(plan.operations)
            if op.eo
                shell_function = if ivdep 
                        :($(radix_family), _shell_y_ivdep!)
                    else
                        :($(radix_family), _shell_y!)
                    end
                push!(ops, :($(shell_function)($(current_input), $(op.stride)))) 
            else
                shell_function = if ivdep
                    :($(radix_family), _shell_ivdep!)
                else
                    :($(radix_family), _shell!)
                end
                push!(ops, :($(shell_function)($(current_output), $(current_input), $(op.stride)))) 
            end
        else
            n1 = op.n_groups ÷ get_radix_divisor(op_type)
            theta = 2 / op.n_groups
            shell_layered_function = if ivdep
                :(radix_family, :_shell_layered_ivdep!)
            else
                :(radix_family, :_shell_layered!)
            end
            push!(ops, :($(shell_layered_function)($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(shell_layered_function($(current_output), $(current_input), $(op.stride), $n1, $theta)))
        end
    end
    =#

    function push_operation!(ops, op_type, op, current_input, current_output, ivdep)
    radix_family = get_radix_family(op_type)
    @show radix_family
    
    # Define suffixes based on conditions
    suffix = if op === last(plan.operations)
        if op.eo
            if ivdep
                :_shell_y_ivdep!
            else
                :_shell_y!
            end
        else
            if ivdep
                :_shell_ivdep!
            else
                :_shell!
            end
        end
    else
        if ivdep
            :_shell_layered_ivdep!
        else
            :_shell_layered!
        end
    end

    # Create the full function name
    shell_function = Symbol(radix_family, suffix)

    if op === last(plan.operations)
        if op.eo
            push!(ops, Expr(:call, shell_function, current_input, op.stride))
        else
            push!(ops, Expr(:call, shell_function, current_output, current_input, op.stride))
        end
    else
        n1 = op.n_groups ÷ get_radix_divisor(op_type)
        theta = 2 / op.n_groups
        push!(ops, Expr(:call, shell_function, current_output, current_input, op.stride, n1, theta))
    end
end

    # A mapping function to determine radix family
    function get_radix_family(op_type::Symbol)
        if startswith(String(op_type), "fft")
            radix = parse(Int, String(op_type)[4:end])
            if radix ∈ [2, 4, 8, 16]
                return :(radix2_family.$(op_type))
            elseif radix ∈ [3, 9]
                return :(radix3_family.$(op_type))
            elseif radix == 5
                return :(radix5_family.$(op_type))
            elseif radix == 7
                return :(radix7_family.$(op_type))
            else
                error("Unsupported radix: $radix")
            end
            return radix_expr
        else
            error("Unknown operation type: $op_type")
        end
    end

    # Divisor mapping for specific FFTs
    function get_radix_divisor(op_type::Symbol)
        radix = parse(Int, String(op_type)[4:end])
        return radix
    end

    # Main loop over operations
    @show plan.operations
    for (i, op) in enumerate(plan.operations)
        if op.op_type ∈ [:fft2, :fft3, :fft4, :fft5, :fft7, :fft8, :fft9, :fft16]
            #Special technical optimization should be considered here:
            if check_ivdep
            radix_family = get_radix_family(op.op_type)
            shell_y_func = op.eo ? get_function_reference(radix_family, :shell_y!) : get_function_reference(radix_family, :shell!)
            shell_y_ivdep_func = get_function_reference(radix_family, :shell_y_ivdep!) : get_function_reference(radix_family, :shell_ivdep!)
            ivdep = determine_ivdep_threshold(
                shell_y_func,
                shell_y_ivdep_func,
                op.n_groups,
                op.stride
            )
            end
            push_operation!(ops, op.op_type, op, current_input, current_output, ivdep)
        else
            error("Unsupported operation type: $(op.op_type)")
        end
        current_input, current_output = current_output, current_input
    end

    # Construct the function body
    function_body = Expr(:block, ops...)

    # Combine all operations into a single runtime-generated function
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)

    show_function && @show ex

    runtime_generated_function = @RuntimeGeneratedFunction(ex)
    return runtime_generated_function
end
=#

function generate_safe_execute_function!(plan::RadixPlan, show_function=true, check_ivdep=false)
    current_input = :x
    current_output = :y
    ops = []
    ivdep = true

    # Helper to get the radix family module and function reference
    function get_radix_family(op_type::Symbol)
        radix = parse(Int, String(op_type)[4:end])
        if radix ∈ [2, 4, 8, 16]
            return radix2_family
        elseif radix ∈ [3, 9]
            return radix3_family
        elseif radix == 5
            return radix5_family
        elseif radix == 7
            return radix7_family
        else
            error("Unsupported radix: $radix")
        end
    end

    function get_function_reference(radix_family, base_function_name::Symbol)
        func = getfield(radix_family, base_function_name)
        if !isdefined(radix_family, base_function_name)
            error("Function $base_function_name not found in module $(radix_family)")
        end
        return func
    end

    # Divisor mapping for specific FFTs
    function get_radix_divisor(op_type::Symbol)
        radix = parse(Int, String(op_type)[4:end])
        return radix
    end

    # Helper to push operations dynamically
    function push_operation!(ops, op, current_input, current_output, ivdep)
        radix_family = get_radix_family(op.op_type)
        suffix = if op === last(plan.operations)
            if op.eo
                ivdep ? :shell_y_ivdep! : :shell_y!
            else
                ivdep ? :shell_ivdep! : :shell!
            end
        else
            ivdep ? :shell_layered_ivdep! : :shell_layered!
        end
        function_name = Symbol(String(op.op_type), "_", String(suffix))
        func_ref = get_function_reference(radix_family, function_name)

        if op === last(plan.operations)
            if op.eo
                push!(ops, Expr(:call, func_ref, current_input, op.stride))
            else
                push!(ops, Expr(:call, func_ref, current_output, current_input, op.stride))
            end
        else
            n1 = op.n_groups ÷ get_radix_divisor(op.op_type)
            theta = 2 / op.n_groups
            push!(ops, Expr(:call, func_ref, current_output, current_input, op.stride, n1, theta))
        end
    end

    # Main loop over operations
    @show plan.operations
    for (i, op) ∈ enumerate(plan.operations)
        if op.op_type ∈ [:fft2, :fft3, :fft4, :fft5, :fft7, :fft8, :fft9, :fft16]
            if check_ivdep
                # Determine if IVDEP is beneficial
                radix_family = get_radix_family(op.op_type)
                base_func_name = if op.eo
                    :fft_shell_y!
                else
                    :fft_shell!
                end
                func_ref = get_function_reference(radix_family, base_func_name)
                ivdep_func_ref = get_function_reference(radix_family, Symbol(base_func_name, "_ivdep!"))
                ivdep = determine_ivdep_threshold(func_ref, ivdep_func_ref, op.n_groups, op.stride)
            end
            push_operation!(ops, op, current_input, current_output, ivdep)
        else
            error("Unsupported operation type: $(op.op_type)")
        end
        current_input, current_output = current_output, current_input
    end

    # Construct the function body
    function_body = Expr(:block, ops...)

    # Combine all operations into a runtime-generated function
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)

    show_function && @show ex

    runtime_generated_function = @RuntimeGeneratedFunction(ex)
    return runtime_generated_function
end

function benchmark_ivdep_performance(fft_standard!, fft_ivdep!, n, s)
    ivdep_speedup_ratios = []
    
    x = [ComplexF64(i,i) for i in 1:n]
    y1 = similar(x)
    y2 = similar(x)
    
    # Warm-up runs
    fft_standard!(y1, x, s)
    fft_ivdep!(y2, x, s)
    
    # Verify correctness
    @assert y1 == y2 "Implementations produce different results!"
    
    # Benchmark standard version
    standard_time = @elapsed begin
        for _ in 1:10
            fft_standard!(y1, x, s)
        end
    end
    
    # Benchmark ivdep version
    ivdep_time = @elapsed begin
        for _ in 1:10
            fft_ivdep!(y2, x, s)
        end
    end
    
    # Calculate speedup ratio
    speedup_ratio = standard_time / ivdep_time
    push!(ivdep_speedup_ratios, speedup_ratio)
    
    println("Array size $n:")
    println("Standard time: $standard_time")
    println("IVDEP time: $ivdep_time")
    println("Speedup ratio: $speedup_ratio")
    
    return ivdep_speedup_ratios
end

function determine_ivdep_threshold(fft_standard!, fft_ivdep!, remaining_n, stride)
    speedup_ratios = benchmark_ivdep_performance(fft_standard!, fft_ivdep!, remaining_n, stride)
    
    # Calculate mean speedup and standard deviation
    mean_speedup = mean(speedup_ratios)
    
    # Determine if IVDEP is consistently beneficial
    is_ivdep_beneficial = mean_speedup > 1.05  # 5% speedup threshold
    
    println("Mean Speedup: $mean_speedup")
    println("IVDEP Beneficial: $is_ivdep_beneficial")
    
    return is_ivdep_beneficial
end

using LoopVectorization

#VERY BAD GENERATOR
#=
function generate_safe_execute_function!(plan::RadixPlan)
    current_input = :x
    current_output = :y
    ops = []
    # ALWAYS TRUE FOR NOW
    ivdep = true

    for (i, op) in enumerate(plan.operations)
        if op.op_type == :fft16
            if op === last(plan.operations)
                if op.eo
                    #ivdep = determine_ivdep_threshold(radix2_family.fft16_shell_y!, radix2_family.fft16_y_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix2_family.fft16_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft16_shell_y!($(current_input), $(op.stride))))
                else
                    #ivdep = determine_ivdep_threshold(radix2_family.fft16_shell!, radix2_family.fft16_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix2_family.fft16_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft16_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                n1 = op.n_groups ÷ 16
                theta = 2 / op.n_groups
                #ivdep = determine_ivdep_threshold(radix2_family.fft16_shell_layered!, radix2_family.fft16_shell_layered_ivdep!, op.n_groups, op.stride)
                ivdep ? push!(ops, :(radix2_family.fft16_shell_layered_ivdep!($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(radix2_family.fft16_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
            end
            elseif op.op_type == :fft9
                if op === last(plan.operations)
                    if op.eo
                        #ivdep = determine_ivdep_threshold(radix3_family.fft9_shell_y!, radix3_family.fft9_y_ivdep!, op.n_groups, op.stride)
                        ivdep = true
                        ivdep ? push!(ops,  :(radix3_family.fft9_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix3_family.fft9_shell_y!($(current_input), $(op.stride))))
                    else
                        ivdep = determine_ivdep_threshold(radix3_family.fft9_shell!, radix3_family.fft9_shell_ivdep!, op.n_groups, op.stride)
                        ivdep ? push!(ops, :(radix3_family.fft9_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix3_family.fft9_shell!($(current_output), $(current_input), $(op.stride))))
                    end
                else
                    n1 = op.n_groups ÷ 9
                    theta = 2 / op.n_groups
                    #ivdep = determine_ivdep_threshold(radix3_family.fft9_shell_layered!, radix3_family.fft9_shell_layered_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops, :(radix3_family.fft9_shell_layered_ivdep!($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(radix3_family.fft9_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
                end
            elseif op.op_type == :fft8
                if op === last(plan.operations)
                if op.eo
                    #ivdep = determine_ivdep_threshold(radix2_family.fft8_shell_y!, radix2_family.fft8_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops, :(radix2_family.fft8_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft8_shell_y!($(current_input), $(op.stride))))
                else
                    ivdep = determine_ivdep_threshold(radix2_family.fft8_shell!, radix2_family.fft8_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix2_family.fft8_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft8_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                    n1 = op.n_groups >> 3
                    theta = 2 / op.n_groups
                    if theta == 0.125
                        #ivdep = determine_ivdep_threshold(radix2_family.fft8_shell_layered_theta_1_8!, radix2_family.fft8_shell_layered_theta_1_8_ivdep!, op.n_groups, op.stride)
                        ivdep = true
                        ivdep ? push!(ops, :(radix2_family.fft8_shell_layered_theta_1_8_ivdep!($(current_output), $(current_input), $(op.stride), $n1))) : push!(ops, :(radix2_family.fft8_shell_layered_theta_1_8!($(current_output), $(current_input), $(op.stride), $n1)))
                    else
                        #ivdep = determine_ivdep_threshold(radix2_family.fft8_shell_layered!, radix2_family.fft8_shell_layered_ivdep!, op.n_groups, op.stride)
                        ivdep = true
                        ivdep ? push!(ops, :(radix2_family.fft8_shell_layered_ivdep!($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(radix2_family.fft8_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
                    end
            end
            elseif op.op_type == :fft7
                if op === last(plan.operations)
                if op.eo
                    #ivdep = determine_ivdep_threshold(radix7_family.fft7_shell_y!, radix7_family.fft7_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops,  :(radix7_family.fft7_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix7_family.fft7_shell_y!($(current_input), $(op.stride))))
                else
                    ivdep = determine_ivdep_threshold(radix7_family.fft7_shell!, radix7_family.fft7_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix7_family.fft7_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix7_family.fft7_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                    n1 = op.n_groups ÷ 7
                    theta = 2 / op.n_groups
                    #ivdep = determine_ivdep_threshold(radix7_family.fft7_shell_layered!, radix7_family.fft7_shell_layered_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops, :(radix7_family.fft7_shell_layered_ivdep!($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(radix7_family.fft7_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
            end
            elseif op.op_type == :fft5
                println("FFT5")
                println(" plan operations $(plan.operations)")
                if op === last(plan.operations)
                    if op.eo
                    #ivdep = determine_ivdep_threshold(radix5_family.fft5_shell_y!, radix5_family.fft5_shell_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops,  :(radix5_family.fft5_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix5_family.fft5_shell_y!($(current_input), $(op.stride))))
                else
                    ivdep = determine_ivdep_threshold(radix5_family.fft5_shell!, radix5_family.fft5_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix5_family.fft5_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix5_family.fft5_shell!($(current_output), $(current_input), $(op.stride))))
                end
            else
                println("op n groups: $(op.n_groups)")
                    n1 = op.n_groups ÷ 5
                    theta = 2 / op.n_groups
                    ivdep = true
                    ivdep ? push!(ops, :(radix5_family.fft5_shell_layered_ivdep!($(current_output), $(current_input), $(op.stride), $n1, $theta))) : push!(ops, :(radix5_family.fft5_shell_layered!($(current_output), $(current_input), $(op.stride), $n1, $theta)))
            end
            elseif op.op_type == :fft4

                if op.eo
                    #ivdep = determine_ivdep_threshold(radix2_family.fft4_shell_y!, radix2_family.fft4_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops,  :(radix2_family.fft4_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix2_family.fft4_shell_y!($(current_input), $(op.stride))))
                else
                    ivdep = determine_ivdep_threshold(radix2_family.fft4_shell!, radix2_family.fft4_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix2_family.fft4_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft4_shell!($(current_output), $(current_input), $(op.stride))))
                end
            elseif op.op_type == :fft3
                    if op.eo
                    #ivdep = determine_ivdep_threshold(radix3_family.fft3_shell_y!, radix3_family.fft3_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops,  :(radix3_family.fft3_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix3_family.fft3_shell_y!($(current_input), $(op.stride))))
                else
                    ivdep = determine_ivdep_threshold(radix3_family.fft3_shell!, radix3_family.fft3_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix3_family.fft3_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix3_family.fft3_shell!($(current_output), $(current_input), $(op.stride))))
                end
            elseif op.op_type == :fft2
                if op.eo
                    #ivdep = determine_ivdep_threshold(radix2_family.fft2_shell_y!, radix2_family.fft2_y_ivdep!, op.n_groups, op.stride)
                    ivdep = true
                    ivdep ? push!(ops,  :(radix2_family.fft2_shell_y_ivdep!($(current_input), $(op.stride)))) : push!(ops,  :(radix2_family.fft2_shell_y!($(current_input), $(op.stride))))
            else
                    ivdep = determine_ivdep_threshold(radix2_family.fft2_shell!, radix2_family.fft2_shell_ivdep!, op.n_groups, op.stride)
                    ivdep ? push!(ops, :(radix2_family.fft2_shell_ivdep!($(current_output), $(current_input), $(op.stride)))) : push!(ops, :(radix2_family.fft2_shell!($(current_output), $(current_input), $(op.stride))))
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

    @show plan.n
    @show ex

    runtime_generated_function = @RuntimeGeneratedFunction(ex)

    return runtime_generated_function
end
=#

end

