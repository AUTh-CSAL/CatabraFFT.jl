module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools

include("radix_3_codelets.jl")
include("radix_5_codelets.jl")
include("radix_7_codelets.jl")
include("helper_tools.jl")

function generate_mat_execute_function!(plan::RadixPlan, show_function=true) 
    T = typeof(plan).parameters[1]
    current_input = :x
    current_output = :y
    ops = []
    ivdep = false
    ivdep_change_exists = false
    check_ivdep = false

    # Helper to push operations dynamically
    function push_operation!(ops, op, current_input, current_output, ivdep)
        suffix = ivdep ? :shell_ivdep! : :shell!
        radix_family = get_radix_family(op.op_type)
        function_name = Symbol(String(op.op_type), "_", String(suffix))
        @show function_name
        func_ref = get_function_reference(radix_family, function_name)

        #if op === last(plan.operations)
            #op.eo ? push!(ops, Expr(:call, func_ref, current_input, op.stride)) : push!(ops, Expr(:call, func_ref, current_output, current_input, op.stride))
        #else
            n1 = op.n_groups ÷ get_radix_divisor(op.op_type)
            theta::T = T(2 / op.n_groups)
            push!(ops, Expr(:call, func_ref, current_output, current_input))
        #end
    end

    # Main loop over operations
    for (i, op) ∈ enumerate(plan.operations)
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
                @show std_func_ref, ivdep_func_ref
                ivdep = determine_ivdep_threshold(std_func_ref, ivdep_func_ref, op, is_layered, typeof(plan).parameters[1] , show_function)
                if ivdep
                    ivdep_change_exists = true
                end
            end
            push_operation!(ops, op, current_input, current_output, ivdep)
        current_input, current_output = current_output, current_input
    end

    # Construct the function body
    function_body = Expr(:block, ops...)

    # Combine all operations into a runtime-generated function
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)
    
    #runtime_generated_function = @RuntimeGeneratedFunction(ex)
    runtime_generated_function = Core.eval(Radix_Execute, ex)
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

function generate_linear_execute_function!(plan::RadixPlan, show_function=true, TECHNICAL_ACCELERATION=true, str="") 
    T = typeof(plan).parameters[1]
    current_input = :x
    current_output = :y
    ops = []
    ivdep = false
    ivdep_change_exists = false
    check_ivdep = TECHNICAL_ACCELERATION

    # Helper to get the radix family module and function reference

    # Helper to push operations dynamically
    function push_operation!(ops, op, current_input, current_output, ivdep)
        radix_family = get_radix_family(op.op_type)
        #=
        suffix = if op === last(plan.operations)
            if op.eo
                ivdep ? :shell_y_ivdep! : :shell_y!
            else
                ivdep ? :shell_ivdep! : :shell!
            end
        else
            ivdep ? :shell_layered_ivdep! : :shell_layered!
        end
        =#
        suffix = ivdep ? :shell_layered_ivdep! : :shell_layered!
        function_name = Symbol(String(op.op_type), "_", String(suffix))
        func_ref = get_function_reference(radix_family, function_name)

        #if op === last(plan.operations)
            #op.eo ? push!(ops, Expr(:call, func_ref, current_input, op.stride)) : push!(ops, Expr(:call, func_ref, current_output, current_input, op.stride))
        #else
            n1 = op.n_groups ÷ get_radix_divisor(op.op_type)
            theta::T = T(2 / op.n_groups)
            push!(ops, Expr(:call, func_ref, current_output, current_input, op.stride, n1, T(theta)))
        #end
    end

    # Main loop over operations
    for (i, op) ∈ enumerate(plan.operations)
        #if op.op_type ∈ [:fft2, :fft3, :fft4, :fft5, :fft7, :fft8, :fft9, :fft16]
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
                @show std_func_ref, ivdep_func_ref
                ivdep = determine_ivdep_threshold(std_func_ref, ivdep_func_ref, op, is_layered, typeof(plan).parameters[1] , show_function)
                if ivdep
                    ivdep_change_exists = true
                end
            end
            push_operation!(ops, op, current_input, current_output, ivdep)
        #else
            #error("Unsupported operation type: $(op.op_type)")
        #end
        current_input, current_output = current_output, current_input
    end

    # Construct the function body
    function_body = Expr(:block, ops...)

    # Combine all operations into a runtime-generated function
    ex = :(function execute_fft_linear!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where T <: AbstractFloat
        $function_body
        return nothing
    end)
    
    #runtime_generated_function = @RuntimeGeneratedFunction(ex)
    runtime_generated_function = Core.eval(Radix_Execute, ex)
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

# Helper function to process the entire function expression
function process_execute_function!(ex::Expr)
    @assert ex.head === :function "Expression must be a function definition"
    
    # Process the function body
    function_body = ex.args[2]
    if function_body isa Expr
        remove_ivdep_suffix!(function_body)
    end
    
    return ex
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

function benchmark_functions_performance(fft_standard!, fft_ivdep!, N::Int, type, show_function=true)
    
    x = randn(Complex{type}, N)
    
    # Warm-up runs
    fft_standard!(x, x) # Recycle random data, I don't care
    fft_ivdep!(x, x)
    
    # Benchmark standard version
    standard_bench = @benchmark $fft_standard!($x, $x) samples=100
    
    # Benchmark ivdep version
    ivdep_bench = @benchmark $fft_ivdep!($x, $x) samples=100

    standard_time = minimum(standard_bench.times)
    ivdep_time = minimum(ivdep_bench.times)

    # Calculate speedup ratio
    speedup_ratio = standard_time / ivdep_time

    # Determine if IVDEP is consistently beneficial
    is_ivdep_beneficial = speedup_ratio > 1.03  # 3% speedup threshold for final functions
    
    if show_function
        println("Final Function Mean Speedup: $(speedup_ratio*100) %")
        println("Final Function IVDEP Beneficial: $is_ivdep_beneficial")
        println("+------------------------------------------------------------+")
    end
    
    return is_ivdep_beneficial
end

function benchmark_ivdep_performance(fft_standard!, fft_ivdep!, op, is_layered, type)
    n = op.n_groups
    s = op.stride
    eo = op.eo
    
    x = randn(Complex{type}, n*s)
    
    if is_layered
        n1 = n ÷ get_radix_divisor(op.op_type)
        theta = 2 / n

        # Warm-up runs
        fft_standard!(x, x, s, n1, theta) # Recycle random data, I don't care
        fft_ivdep!(x, x, s, n1, theta)
    
        # Benchmark standard version
        #standard_bench = @benchmark $fft_standard!($x, $x, $s, $n1, $theta) samples=100
        standard_time = time_limited_benchmark(fft_standard!, x, x, s, n1, theta)
    
        # Benchmark ivdep version
        #ivdep_bench = @benchmark $fft_ivdep!($x, $x, $s, $n1, $theta) samples=100
        ivdep_time = time_limited_benchmark(fft_ivdep!, x, x, s, n1, theta)

    else
        if eo

        fft_standard!(x, s) # Recycle random data, I don't care
        fft_ivdep!(x, s)
        
        # Benchmark standard version
        #standard_bench = @benchmark $fft_standard!($x, $s) samples=100
        standard_time = time_limited_benchmark(fft_standard!, x, s)
        
        # Benchmark ivdep version
        #ivdep_bench = @benchmark $fft_ivdep!($x, $s) samples=100
        ivdep_time = time_limited_benchmark(fft_ivdep!, x, s)
    else
        # Warm-up runs
        fft_standard!(x, x, s) # Recycle random data, I don't care
        fft_ivdep!(x, x, s)
        
        # Benchmark standard version
        #standard_bench = @benchmark $fft_standard!($x, $x, $s) samples=100
        standard_time = time_limited_benchmark(fft_standard!, x, x, s)
        
        # Benchmark ivdep version
        #ivdep_bench = @benchmark $fft_ivdep!($x, $x, $s) samples=100
        ivdep_time = time_limited_benchmark(fft_ivdep!, x, x, s)

    end
end

    #standard_time = minimum(standard_bench.times)
    #ivdep_time = minimum(ivdep_bench.times)
    # Calculate speedup ratio
    speedup_ratio = standard_time / ivdep_time
    
    return speedup_ratio
end

function determine_ivdep_threshold(fft_standard!, fft_ivdep!, op, is_layered, type, show_function)
    speedup_ratio = benchmark_ivdep_performance(fft_standard!, fft_ivdep!, op, is_layered, type)
    
    # Determine if IVDEP is consistently beneficial
    is_ivdep_beneficial = speedup_ratio > 1.05  # 5% speedup threshold for single codelets
    
    if show_function
        println("-> Shell $(op.op_type) n = $(op.n_groups) s = $(op.stride) Mean Speedup: $(speedup_ratio*100) %")
        println("-> Shell $(op.op_type) IVDEP Beneficial: $is_ivdep_beneficial")
    end
    
    return is_ivdep_beneficial
end

# Divisor mapping for specific FFTs
function get_radix_divisor(op_type::Symbol)
    radix = parse(Int, String(op_type)[4:end])
    return radix
end

# Benchmarking different theoretical plans used by the MEASURE >= FLAGS
#dummy function
function return_best_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool, TECHNICAL_ACCELERATION::Bool) where T <: AbstractFloat
    N = plans[1].n
    best_time = Inf
    best_func = nothing
    x = randn(Complex{T}, N)
    evaluate_fft_generated_module(Radix_Execute, N, T)
    
    for plan in plans
        test_func = generate_linear_execute_function!(plan, true, TECHNICAL_ACCELERATION) 
        show_function && println("Testing function for plan: $plan")
        
        #Pre-compute
        #test_func(x,x)
        Base.invokelatest(test_func, x, x)
        
        # Use @belapsed for faster benchmarking with minimal overhead
        test_time = @elapsed Base.invokelatest(test_func, x, x) 
        #test_time = @benchmark Base.invokelatest(test_func, x, x) 
        
        println("Test elapsed time: $test_time seconds")
        
        if test_time < best_time
            best_func = test_func
            best_time = test_time
        end
    end
    
    show_function && println("Best function: $best_func with time: $best_time seconds")
    
    return best_func
end

function return_best_static_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool, TECHNICAL_ACCELERATION::Bool) where T <: AbstractFloat
    N = plans[1].n
    best_time = Inf
    best_func = nothing
    x = rand(Complex{T}, N)
    
    for plan in plans
        println("Creating new module")
        evaluate_fft_generated_module(Radix_Execute, plan, T)
        test_func = generate_mat_execute_function!(plan, true)
        show_function && println("Testing module for plan: $plan")
        
        Base.invokelatest(test_func, x, x)
        
        test_time = @elapsed Base.invokelatest(test_func, x, x) 
        
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