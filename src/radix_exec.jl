
module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools

include("helper_tools.jl")

# Generate a complete monolithic FFT function with all kernels inlined
function generate_mat_execute_function!(plan::RadixPlan, show_function=false)
    T = typeof(plan).parameters[1]
    
    # Extract kernel expressions directly
    kernel_exprs = extract_kernel_expressions(plan, T)
    
    show_function && println("Available kernels: ", collect(keys(kernel_exprs)))
    
    # Build the complete execution function with inlined kernels
    ops = Expr[]
    
    # Extract constants once at the beginning - using Dict to avoid duplicates
    constants_dict = extract_constants_dict(kernel_exprs)
    for (const_name, const_value) in constants_dict
        push!(ops, :($(Symbol(const_name)) = $const_value))
    end
    
    # Track buffer state for Stockham algorithm
    current_input = :x
    current_output = :y
    
    for (stage_idx, op) in enumerate(plan.operations)
        is_final_stage = (stage_idx == length(plan.operations))
        radix = get_radix_divisor(op.op_type)
        n_g = op.n_groups
        stride = op.stride
        SIZE = n_g * stride
        
        show_function && println("Stage $stage_idx: radix=$radix, n_groups=$n_g, stride=$stride, input=$current_input, output=$current_output")
        
        if !is_final_stage
            # Non-final stage: inline radix kernels for each group
            n_groups_per_radix = SIZE ÷ radix  # Total elements divided by radix
            
            show_function && println("  n_groups_per_radix = $n_groups_per_radix")
            
            for p in 0:(n_groups_per_radix-1)
                kernel_key = "fft$(radix)_$(stride)x$(n_g)_$(p)!"
                show_function && println("  Looking for kernel: $kernel_key")
                
                if haskey(kernel_exprs, kernel_key)
                    kernel_body = kernel_exprs[kernel_key]
                    # Remove constants that are already defined
                    kernel_body = remove_constants_from_kernel(kernel_body)
                    # Substitute input/output variables
                    kernel_body = substitute_kernel_vars(kernel_body, current_output, current_input)
                    push!(ops, kernel_body)
                    show_function && println("    ✓ Found and added kernel")
                else
                    show_function && println("    ✗ Kernel not found")
                    error("Missing kernel: $kernel_key")
                end
            end
        else
            # Final stage: use direct indexing instead of views
            for j in 1:stride
                kernel_key = "fft$(radix)_$(stride)x$(n_g)_0!"
                show_function && println("  Looking for final kernel: $kernel_key, stride index: $j")
                
                if haskey(kernel_exprs, kernel_key)
                    kernel_body = kernel_exprs[kernel_key]
                    # Remove constants
                    kernel_body = remove_constants_from_kernel(kernel_body)
                    # Replace with direct strided indexing
                    kernel_body = substitute_strided_final_stage(kernel_body, current_output, current_input, j, stride, SIZE)
                    push!(ops, kernel_body)
                    show_function && println("    ✓ Found and added final kernel with direct indexing")
                else
                    show_function && println("    ✗ Final kernel not found")
                    error("Missing kernel: $kernel_key")
                end
            end
        end
        
        # Stockham buffer swapping for next stage
        current_input, current_output = current_output, current_input
    end
    
    # Check if we need a final copy
    # After all stages, the result should be in y
    # current_input now points to where the result is
    #=
    if current_input != :y
        show_function && println("Adding final copy from $current_input to y")
        push!(ops, :(copyto!(y, x)))
    end
    =#
    
    # Build the complete monolithic function
    function_body = if isempty(ops)
        :(copyto!(y, x))
    else
        Expr(:block, ops...)
    end
    
    # Generate as a pure Julia function
    func_expr = quote
        function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @inbounds begin
                $function_body
            end
            return nothing
        end
    end
    
    show_function && println("Generated monolithic FFT function with $(length(ops)) operations. \n $func_expr")
    
    # Evaluate in the current module context
    #return Core.eval(@__MODULE__, func_expr)
    Core.eval(@__MODULE__, func_expr)
    return (y, x) -> Base.invokelatest(eval(func_expr, y, x))
end

# Extract constants as dictionary to avoid duplicates
function extract_constants_dict(kernel_exprs::Dict{String, Expr})
    constants_dict = Dict{String, Any}()
    
    for (name, expr) in kernel_exprs
        extract_constants_recursive!(constants_dict, expr)
    end
    
    return constants_dict
end

function extract_constants_recursive!(constants::Dict{String, Any}, expr)
    if isa(expr, Expr)
        if expr.head == :(=) && length(expr.args) == 2
            lhs = expr.args[1]
            rhs = expr.args[2]
            if isa(lhs, Symbol)
                lhs_str = string(lhs)
                if lhs_str == "INV_SQRT2" || 
                   startswith(lhs_str, "COSPI_") || 
                   startswith(lhs_str, "SINPI_")
                    # Store only if not already present
                    if !haskey(constants, lhs_str)
                        constants[lhs_str] = rhs
                    end
                end
            end
        elseif expr.head == :block
            for arg in expr.args
                if isa(arg, Expr)
                    extract_constants_recursive!(constants, arg)
                end
            end
        else
            for arg in expr.args
                if isa(arg, Expr)
                    extract_constants_recursive!(constants, arg)
                end
            end
        end
    end
end

# Remove constants from individual kernels
function remove_constants_from_kernel(expr::Expr)
    return postwalk(expr) do ex
        if isa(ex, Expr)
            if ex.head == :(=) && length(ex.args) == 2
                lhs = ex.args[1]
                if isa(lhs, Symbol)
                    lhs_str = string(lhs)
                    if lhs_str == "INV_SQRT2" || 
                       startswith(lhs_str, "COSPI_") || 
                       startswith(lhs_str, "SINPI_")
                        # Return nothing to signal removal
                        return nothing
                    end
                end
            elseif ex.head == :block
                # Filter out nothing values
                filtered_args = []
                for arg in ex.args
                    walked = postwalk(identity, arg)  # Process recursively
                    if !isnothing(walked)
                        # Check if it's a constant assignment
                        if isa(walked, Expr) && walked.head == :(=) && length(walked.args) == 2
                            lhs = walked.args[1]
                            if isa(lhs, Symbol)
                                lhs_str = string(lhs)
                                if !(lhs_str == "INV_SQRT2" || 
                                     startswith(lhs_str, "COSPI_") || 
                                     startswith(lhs_str, "SINPI_"))
                                    push!(filtered_args, walked)
                                end
                            else
                                push!(filtered_args, walked)
                            end
                        else
                            push!(filtered_args, walked)
                        end
                    end
                end
                return isempty(filtered_args) ? nothing : Expr(:block, filtered_args...)
            end
        end
        return ex
    end
end

# Substitute variables for final stage with direct strided indexing
function substitute_strided_final_stage(kernel_expr::Expr, out_var, in_var, offset::Int, stride::Int, size::Int)
    return postwalk(kernel_expr) do ex
        if isa(ex, Expr) && ex.head == :ref
            if length(ex.args) >= 2
                array_name = ex.args[1]
                index_expr = ex.args[2]
                
                # For the final stage kernel, y[1] becomes out_var[offset], y[2] becomes out_var[offset + stride]
                if array_name == :y
                    if isa(index_expr, Int)
                        actual_index = offset + (index_expr - 1) * stride
                        return Expr(:ref, out_var, actual_index)
                    else
                        # Handle symbolic indices
                        return Expr(:ref, out_var, :($offset + ($index_expr - 1) * $stride))
                    end
                elseif array_name == :x
                    if isa(index_expr, Int)
                        actual_index = offset + (index_expr - 1) * stride
                        return Expr(:ref, in_var, actual_index)
                    else
                        return Expr(:ref, in_var, :($offset + ($index_expr - 1) * $stride))
                    end
                end
            end
        elseif ex == :y
            # This shouldn't happen in well-formed kernels, but handle it
            return out_var
        elseif ex == :x
            return in_var
        elseif isa(ex, Symbol)
            # Replace y1, y2, etc. with appropriate variable names
            s = string(ex)
            if startswith(s, "y") && length(s) > 1
                # Parse the number after 'y'
                num_str = s[2:end]
                if all(isdigit, num_str)
                    idx = parse(Int, num_str)
                    # Generate proper variable name: y1 -> y1, y5, etc.
                    actual_index = offset + (idx - 1) * stride
                    return Symbol("y$actual_index")
                end
            elseif startswith(s, "x") && length(s) > 1
                num_str = s[2:end]
                if all(isdigit, num_str)
                    idx = parse(Int, num_str)
                    actual_index = offset + (idx - 1) * stride
                    return Symbol("x$actual_index")
                end
            end
        end
        return ex
    end
end


# Standard variable substitution for non-final stages
function substitute_kernel_vars(kernel_expr::Expr, out_var, in_var)
    return postwalk(kernel_expr) do ex
        if ex == :y
            return out_var
        elseif ex == :x
            return in_var
        elseif isa(ex, Expr) && ex.head == :ref
            if length(ex.args) >= 2
                if ex.args[1] == :y
                    return Expr(:ref, out_var, ex.args[2:end]...)
                elseif ex.args[1] == :x
                    return Expr(:ref, in_var, ex.args[2:end]...)
                end
            end
        end
        return ex
    end
end

# Extract kernel expressions directly
function extract_kernel_expressions(plan::RadixPlan{T}, ::Type{T}) where T
    plan_data = (n=plan.n, operations=plan.operations)
    return RadixGenerator.create_kernel_module(plan_data, T)
end

# Simple expression tree walker
function postwalk(f, expr)
    if isa(expr, Expr)
        new_args = []
        for arg in expr.args
            walked = postwalk(f, arg)
            if !isnothing(walked)  # Skip nothing values
                push!(new_args, walked)
            end
        end
        return f(Expr(expr.head, new_args...))
    else
        return f(expr)
    end
end

# Benchmarking functions
function return_best_static_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool) where T <: AbstractFloat
    N = plans[1].n
    best_time = Inf
    best_func = nothing
    x = rand(Complex{T}, N)
    y = similar(x)
    
    for plan in plans
        try
            test_func = generate_mat_execute_function!(plan, show_function)
            
            show_function && println("Testing plan: $(plan.operations)")
            
            # Warmup
            test_func(y, x)
            
            # Benchmark
            test_time = time_limited_benchmark(test_func, y, x)
            
            show_function && println("Plan time: $test_time seconds")
            
            if test_time < best_time
                best_func = test_func
                best_time = test_time
            end
            
        catch e
            @warn "Failed to generate/benchmark plan $(plan.operations): $e"
            continue
        end
    end

    show_function && println("Best plan time: $best_time seconds")
    
    if best_func === nothing
        @warn "All plans failed, generating fallback"
        return generate_fallback_function(N, T)
    end
    
    return best_func
end

function generate_fallback_function(n::Int, ::Type{T}) where T
    func_expr = quote
        function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @inbounds for k in 1:$n
                y[k] = zero(Complex{$T})
                for j in 1:$n
                    twiddle = cispi($T(-2) * (k-1) * (j-1) / $n)
                    y[k] += x[j] * twiddle
                end
            end
            return nothing
        end
    end
    
    return Core.eval(@__MODULE__, func_expr)
end

function time_limited_benchmark(f, y, x; time_limit=0.05)
    f(y, x)  # Warmup
    
    total_time = 0.0
    count = 0
    
    while total_time < time_limit && count < 100
        elapsed_time = @elapsed f(y, x)
        total_time += elapsed_time
        count += 1
    end
    
    return count > 0 ? total_time / count : Inf
end

function return_best_linear_function(plans::Vector{RadixPlan{T}}, show_function::Bool, ivdep::Bool) where T <: AbstractFloat
    return return_best_static_linear_function(plans, show_function)
end

function generate_linear_execute_function!(plan::RadixPlan, show_function::Bool, ivdep::Bool)
    return generate_mat_execute_function!(plan, show_function)
end

end