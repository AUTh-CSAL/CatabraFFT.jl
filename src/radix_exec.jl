
module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools
using SIMD

include("helper_tools.jl")

# Generate a complete monolithic FFT function or function group for decompositions with all kernels inlined
function GenerateMatrixExpr!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]

    # 1) Gather kernels
    plan_data = (n=plan.n, operations=plan.operations)
    kernel_exprs = RadixGenerator.create_kernel_dictionary(plan_data, T)

    show_function && println("Available kernels: ", collect(keys(kernel_exprs)))

    # 2) Collect constants once
    ops = Expr[]
    constants_dict = extract_constants_dict(kernel_exprs)
    for (const_name, const_value) in constants_dict
        push!(ops, :($(Symbol(const_name)) = $const_value))
    end


    for (stage_idx, op) in enumerate(plan.operations)

        if !op.eo
            current_input  = :x
            current_output = :y
        else
            current_input  = :y
            current_output = :x
        end

        is_final_stage = (stage_idx == length(plan.operations))
        radix  = get_radix_divisor(op.op_type)
        n_g    = op.n_groups
        stride = op.stride
        SIZE   = n_g * stride
        is_monolithic_shell = (radix == n_g) && (stride == 1)
        
        show_function && println("Stage $stage_idx: radix=$radix, n_groups=$n_g, stride=$stride, in=$current_input, out=$current_output")

        if is_monolithic_shell
            key = "fft$(radix)_shell!"
            show_function && println("  kernel: $key")
            haskey(kernel_exprs, key) || error("Missing kernel: $key")
            body = kernel_exprs[key]
            push!(ops, body)
        elseif !is_final_stage 
            n_groups_per_radix = SIZE ÷ radix
            
            for p in 0:(n_groups_per_radix-1)
                # Each kernel has unique p and hardcoded indices
                key = "fft$(radix)_$(stride)x$(n_g)_$(p)!"
                
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                body = kernel_exprs[key]
                #body = remove_constants_from_kernel(body)
                #body = substitute_kernel_vars(body, current_output, current_input)
                #show_function && println("Sub-Kernel Named $key with Body: $body")
                show_function && println("Sub-Kernel Named $key with Body: ")
                push!(ops, body)
            end
        else
            # Terminal stage: generate loop-based kernel
            key = "fft$(radix)_$(stride)x$(n_g)_0!"
            haskey(kernel_exprs, key) || error("Missing kernel: $key")
            body = kernel_exprs[key]
            
            #show_function && println("Terminal Kernel Named $key (looped $stride times) with Body: $body")
            show_function && println("Terminal Kernel Named $key (looped $stride times) with Body: ")
            
            # Transform template to use idx variable
            if length(plan.operations) % 2 == 0 current_output = current_input end # has_y condition
            loop_body = substitute_strided_final_loop(body, current_output, current_input, stride, SIZE, radix)
            
            # Wrap in loop
            loop_expr = quote
                @inbounds @simd ivdep for idx in 1:$stride
                    $loop_body
                end
            end
            
            push!(ops, loop_expr)
        end
        # Stockham swap
        #current_input, current_output = current_output, current_input
    end

    # 4) Final body block
    BLOCK = Expr(:block, ops...)
    return BLOCK
    #return isempty(ops) ? :(copyto!(y, x)) : 
end

function generate_mat_execute_function!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]
    function_body = GenerateMatrixExpr!(plan, show_function)
    
    show_function && println("Function body: \n $function_body")

    func_expr = quote
        @inline function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @fastmath @inbounds begin
                $function_body
            end
            nothing
        end
    end

    show_function && println("Generated monolithic FFT function.\n$func_expr")
    return func_expr
end

function materialize_plan_function!(plan::RadixPlan, ::Type{T}) where {T}
    constants_dict = RadixGenerator.generate_local_constants_dict(plan.n, T)
    body = GenerateMatrixExpr!(plan, false)
    substituted_body = substitute_constants_in_expr(body, constants_dict)
    
    fexpr = quote
        @inline function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @fastmath @inbounds begin
                $substituted_body
            end
            nothing
        end
    end
    
    # Clean display without line numbers
    # clean_expr = Base.remove_linenums!(deepcopy(fexpr))
    # @show clean_expr

    return eval(fexpr)
end

# Function to substitute constant symbols with literal values
function substitute_constants_in_expr(expr, constants_dict::Dict{Symbol, T}) where T
    if isa(expr, Expr)
        # Recursively process all sub-expressions
        new_args = [substitute_constants_in_expr(arg, constants_dict) for arg in expr.args]
        return Expr(expr.head, new_args...)
    elseif isa(expr, Symbol)
        # Replace symbol with literal value if it's a constant
        if haskey(constants_dict, expr)
            return constants_dict[expr]
        else
            return expr  # Keep symbol as-is if not a constant
        end
    else
        return expr  # Return literals unchanged
    end
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

# Corrected substitute_strided_final_loop function
# This function is CORRECT - it transforms p=0 kernel template for looping
function substitute_strided_final_loop(kernel_expr::Expr, out_var, in_var, stride::Int, size::Int, radix::Int)
    input_spacing = size ÷ radix
    
    # Normalize variable names to Symbols for comparison
    out_sym = out_var isa Symbol ? out_var : Symbol(out_var)
    in_sym = in_var isa Symbol ? in_var : Symbol(in_var)
    
    # Simple recursive walk
    function transform(ex, is_output_context::Bool)
        if isa(ex, Expr)
            if ex.head == :ref && length(ex.args) == 2
                arr = ex.args[1]
                idx = ex.args[2]
                
                # Normalize array name to Symbol
                arr_sym = arr isa Symbol ? arr : Symbol(arr)
                
                # Check for OUTPUT array references
                if is_output_context && arr_sym == out_sym && isa(idx, Int)
                    # Output transformation: y[k] → y[idx + (k-1)*stride]
                    # Example with stride=4:
                    #   y[1] → y[idx]
                    #   y[2] → y[idx + 4]
                    offset = (idx - 1) * stride
                    return offset == 0 ? Expr(:ref, out_sym, :idx) : Expr(:ref, out_sym, :(idx + $offset))
                    
                # Check for INPUT array references
                elseif !is_output_context && arr_sym == in_sym && isa(idx, Int)
                    # Input transformation: x[k] → x[idx + (k-1)]
                    # The input spacing was already baked into the template indices
                    # Example: template has x[1], x[5] which become x[idx], x[idx+4]
                    offset = idx - 1
                    return offset == 0 ? Expr(:ref, in_sym, :idx) : Expr(:ref, in_sym, :(idx + $offset))
                end
                
            elseif ex.head == :(=)
                # Assignment: LHS writes to output, RHS reads from input/temps
                lhs = ex.args[1]
                rhs = ex.args[2]
                
                new_lhs = transform(lhs, true)   # LHS is output context
                new_rhs = transform(rhs, false)  # RHS is input context
                return Expr(:(=), new_lhs, new_rhs)
            else
                # Recursively transform all args, preserving context
                new_args = [transform(arg, is_output_context) for arg in ex.args]
                return Expr(ex.head, new_args...)
            end
        end
        return ex
    end
    
    return transform(kernel_expr, false)
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
function return_best_static_linear_expr(plans::Vector{RadixPlan{T}}, show_function::Bool)::Expr where T<:AbstractFloat
    @assert !isempty(plans)
    N = plans[1].n

    # fixed inputs for fair timing
    x = rand(Complex{T}, N)
    y = similar(x)

    best_time = Inf
    best_body_expr::Union{Expr,Nothing} = nothing
    
    for plan in plans
        try
            show_function && println("Benchmarking plan: ", plan.operations)

            f = materialize_plan_function!(plan, T)  # install callable
            
            show_function && println("Materialized function $(plan.operations)")
            # warmup
            Base.invokelatest(f, y, x)

            t = @belapsed Base.invokelatest($f, $y, $x)
            
            show_function && println("Benchmarked time: $t of plan: $(plan.operations)")

            if t < best_time
                best_time = t
                best_body_expr = GenerateMatrixExpr!(plan, show_function)  # store BODY expr for compile-time splice
            end
        catch e
            @warn "Failed to benchmark plan $(plan.operations): $e"
        end
    end
    
    show_function && println("Best time: $best_time of plan: $best_body_expr")

    best_body_expr === nothing && error("No valid plan found")
    return best_body_expr
end

function generate_linear_execute_function!(plan::RadixPlan, show_function::Bool, ivdep::Bool)
    return GenerateMatrixExpr!(plan, show_function)
end

end
