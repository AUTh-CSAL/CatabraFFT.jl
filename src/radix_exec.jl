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



function generate_safe_execute_function!(plan::RadixPlan, show_function=false, check_ivdep=true)
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
    for (i, op) ∈ enumerate(plan.operations)
        if op.op_type ∈ [:fft2, :fft3, :fft4, :fft5, :fft7, :fft8, :fft9, :fft16]
            if check_ivdep
                # Determine if IVDEP is beneficial
                radix_family = get_radix_family(op.op_type)
                suffix, is_layered = if op === last(plan.operations)
                    if op.eo
                        :shell_y, false
                    else
                        :shell, false
                    end
                else
                    :shell_layered, true
                end
                std_func_name = Symbol(String(op.op_type), "_", String(suffix), "!")
                ivdep_func_name = Symbol(String(op.op_type), "_", String(suffix), "_ivdep!")
                std_func_ref = get_function_reference(radix_family, std_func_name)
                ivdep_func_ref = get_function_reference(radix_family, ivdep_func_name)
                ivdep = determine_ivdep_threshold(std_func_ref, ivdep_func_ref, op, is_layered, show_function)

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

function benchmark_ivdep_performance(fft_standard!, fft_ivdep!, op, is_layered)
    ivdep_speedup_ratios = []
    n = op.n_groups
    s = op.stride
    eo = op.eo
    
    x = randn(ComplexF64, n*s)
    
    if is_layered
        n1 = n ÷ get_radix_divisor(op.op_type)
        theta = 2 / n

        # Warm-up runs
        fft_standard!(x, x, s, n1, theta) # Recycle random data, I don't care
        fft_ivdep!(x, x, s, n1, theta)
    
        # Benchmark standard version
        standard_time = @elapsed begin
        for _ in 1:10
            fft_standard!(x, x, s, n1, theta)
        end
        end
    
        # Benchmark ivdep version
        ivdep_time = @elapsed begin
        for _ in 1:10
            fft_ivdep!(x, x, s, n1, theta)
        end
        end

    else
        if eo

        fft_standard!(x, s) # Recycle random data, I don't care
        fft_ivdep!(x, s)
        
        
        # Benchmark standard version
        standard_time = @elapsed begin
            for _ in 1:10
                fft_standard!(x, s)
            end
        end
        
        # Benchmark ivdep version
        ivdep_time = @elapsed begin
            for _ in 1:10
                fft_ivdep!(x, s)
            end
        end
    else
        
    # Warm-up runs
    fft_standard!(x, x, s) # Recycle random data, I don't care
    fft_ivdep!(x, x, s)
    
    
    # Benchmark standard version
    standard_time = @elapsed begin
        for _ in 1:10
            fft_standard!(x, x, s)
        end
    end
    
    # Benchmark ivdep version
    ivdep_time = @elapsed begin
        for _ in 1:10
            fft_ivdep!(x, x, s)
        end
    end
end
end
    
    # Calculate speedup ratio
    speedup_ratio = standard_time / ivdep_time
    push!(ivdep_speedup_ratios, speedup_ratio)
    
    return ivdep_speedup_ratios
end

function determine_ivdep_threshold(fft_standard!, fft_ivdep!, op, is_layered, show_function)
    speedup_ratios = benchmark_ivdep_performance(fft_standard!, fft_ivdep!, op, is_layered)
    
    # Calculate mean speedup and standard deviation
    mean_speedup = mean(speedup_ratios)
    
    # Determine if IVDEP is consistently beneficial
    is_ivdep_beneficial = mean_speedup > 1.05  # 5% speedup threshold
    
    if show_function
    println("Mean Speedup: $(mean_speedup*100) %")
    println("IVDEP Beneficial: $is_ivdep_beneficial")
    end
    
    return is_ivdep_beneficial
end

# Divisor mapping for specific FFTs
function get_radix_divisor(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    return radix
end


end

