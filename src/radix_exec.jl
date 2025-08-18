module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools, RuntimeGeneratedFunctions

include("helper_tools.jl")

RuntimeGeneratedFunctions.init(@__MODULE__)

function generate_mat_execute_function!(plan::RadixPlan, show_function=true)
    T = typeof(plan).parameters[1]
    current_input = :x
    current_output = :y
    ops = []
    ivdep = false
    ivdep_change_exists = false
    check_ivdep = false

    # Generalized operation handler
    function push_radix_operation!(op, future_op)
        radix = get_radix_divisor(op.op_type)  # Returns 8, 4, 2 etc.
        suffix = :shell!
        func_base = Symbol("fft$(radix)")
        n_g, stride = op.n_groups, op.stride
        SIZE = n_g * stride
        
        # Generate kernel calls with dynamic unrolling
        if radix != plan.n && !isnothing(future_op)

            # Generate loop with proper SIMD structure
            loop_var = gensym("i")
            loop_body = Expr(:block)
            for i in 0:op.n_groups-1
                kernel = Symbol(func_base, "_$(stride)x$(n_g)_$(i)!")
                push!(loop_body.args,
                    :(radix_2_family.$kernel($current_output, $current_input)))
            end
            push!(ops, loop_body)

        elseif radix == plan.n
            radix_family = get_radix_family(op.op_type)
            function_name = Symbol(func_base, "_shell!")
            function_ref = get_function_reference(radix_family, function_name)
            push!(ops, Expr(:call, function_ref, current_output, current_input))

        elseif isnothing(future_op) # Last of decomposition calls of mixed-radix call
        loop_var = gensym("_")
        loop_body = Expr(:block)
            kernel = Symbol(func_base, "_$(stride)x$(n_g)_0!")
            push!(loop_body.args,
                :(radix_2_family.$kernel(view($current_input, ($loop_var):$stride:$SIZE))))
        
        loop_iteration = Expr(:(=), loop_var, 1:(op.stride))

        # Build complete loop expression
        loop_expr = Expr(:macrocall,
            Symbol("@inbounds"),
            LineNumberNode(@__LINE__, Symbol(@__FILE__)),
            ivdep ? Expr(:macrocall,
                Symbol("@simd"),
                LineNumberNode(@__LINE__, Symbol(@__FILE__)),
                :ivdep,
                Expr(:for, loop_iteration, loop_body)
            ) :
            Expr(:macrocall,
                Symbol("@simd"),
                LineNumberNode(@__LINE__, Symbol(@__FILE__)),
                Expr(:for, loop_iteration, loop_body)
            )
        )
        push!(ops, loop_expr)

        end
    end

    # Main processing loop
    for (i, op) in enumerate(plan.operations)
        future_op = i < length(plan.operations) ? plan.operations[i+1] : nothing
        push_radix_operation!(op, future_op)
        current_input, current_output = current_output, current_input  # Swap buffers
    end

    # Final function assembly
    function_body = Expr(:block, ops...)
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, 
        x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)

    @show ex
    
    runtime_generated_function = @RuntimeGeneratedFunction(ex)
    if check_ivdep && ivdep_change_exists
        # Create a similar function with ivdep turned off to compare
        clean_generated_function = generate_linear_execute_function!(plan, true, false, "CLEAN")
        
        if !benchmark_functions_performance(clean_generated_function, runtime_generated_function , plan.n, typeof(plan).parameters[1], show_function)
            show_function && println("NON-IVDEP FUNCTION IS BETTER")
            runtime_generated_function = clean_generated_function
        end
    end
    return runtime_generated_function
end

# Helper function to bench via @elapsed
function time_limited_benchmark(f, args...; time_limit=0.1)
    total_time = 0.0
    count = 0
    result = 0.0
    while total_time < time_limit
        # Precompute
        f(args...)
        elapsed_time = @elapsed f(args...)
        total_time += elapsed_time
        result += elapsed_time
        count += 1
    end
    return result / count
end

#=
function return_best_static_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool) where T <: AbstractFloat
    N = plans[1].n
    best_time = Inf
    best_func = nothing
    x = rand(Complex{T}, N)
    
    for plan in plans
        evaluate_fft_generated_module(Radix_Execute, plan, T) # CREATE ALL KERNEL PARTS
        test_func = generate_mat_execute_function!(plan, true) # CONSTRUCT THEM AS A SIGNLE FUNCTION
        show_function && println("Testing module for plan: $plan")
        
        Base.invokelatest(test_func, x, x)
        
        test_time = time_limited_benchmark(@elapsed Base.invokelatest(test_func, x, x))
        
        println("Test elapsed time: $test_time seconds")
        
        if test_time < best_time
            best_func = test_func
            best_time = test_time
        end
    end

    show_function && println("Best function: $best_func with time: $best_time seconds")
    
    return best_func
end
=#

function return_best_static_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool) where T <: AbstractFloat
    N = plans[1].n
    best_time = Inf
    best_func = nothing
    x = rand(Complex{T}, N)
    
    for plan in plans
        evaluate_fft_generated_module(Radix_Execute, plan, T) # CREATE ALL KERNEL PARTS
        test_func = generate_mat_execute_function!(plan, true) # CONSTRUCT THEM AS A SIGNLE FUNCTION
        show_function && println("Testing module for plan: $plan")
        
        Base.invokelatest(test_func, x, x)
        
        # Fixed line: pass a function instead of @elapsed result
        test_time = time_limited_benchmark(() -> Base.invokelatest(test_func, x, x))
        
        println("Test elapsed time: $test_time seconds")
        
        if test_time < best_time
            best_func = test_func
            best_time = test_time
        end
    end

    show_function && println("Best function: $best_func with time: $best_time seconds")
    
    return best_func
end

end