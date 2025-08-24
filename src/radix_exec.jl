
module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools

include("helper_tools.jl")

# Generate a complete monolithic FFT function with all kernels inlined
function generate_mat_execute_expr!(plan::RadixPlan, show_function::Bool=true)::Expr
    T = typeof(plan).parameters[1]

    # 1) Gather kernels

    # 1) Gather kernels
    kernel_exprs = extract_kernel_expressions(plan, T)
    show_function && println("Available kernels: ", collect(keys(kernel_exprs)))

    # 2) Collect constants once

    # 2) Collect constants once
    ops = Expr[]
    constants_dict = extract_constants_dict(kernel_exprs)
    for (const_name, const_value) in constants_dict
        push!(ops, :($(Symbol(const_name)) = $const_value))
    end

    # 3) Inline all stages
    current_input  = :x

    # 3) Inline all stages
    current_input  = :x
    current_output = :y


    for (stage_idx, op) in enumerate(plan.operations)
        is_final_stage = (stage_idx == length(plan.operations))
        radix  = get_radix_divisor(op.op_type)
        n_g    = op.n_groups
        radix  = get_radix_divisor(op.op_type)
        n_g    = op.n_groups
        stride = op.stride
        SIZE   = n_g * stride

        show_function && println("Stage $stage_idx: radix=$radix, n_groups=$n_g, stride=$stride, in=$current_input, out=$current_output")

        SIZE   = n_g * stride

        show_function && println("Stage $stage_idx: radix=$radix, n_groups=$n_g, stride=$stride, in=$current_input, out=$current_output")

        if !is_final_stage
            n_groups_per_radix = SIZE ÷ radix
            n_groups_per_radix = SIZE ÷ radix
            for p in 0:(n_groups_per_radix-1)
                key = "fft$(radix)_$(stride)x$(n_g)_$(p)!"
                show_function && println("  kernel: $key")
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                body = kernel_exprs[key]
                body = remove_constants_from_kernel(body)
                body = substitute_kernel_vars(body, current_output, current_input)
                push!(ops, body)
                key = "fft$(radix)_$(stride)x$(n_g)_$(p)!"
                show_function && println("  kernel: $key")
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                body = kernel_exprs[key]
                body = remove_constants_from_kernel(body)
                body = substitute_kernel_vars(body, current_output, current_input)
                push!(ops, body)
            end
        else
            for j in 1:stride
                key = "fft$(radix)_$(stride)x$(n_g)_0!"
                show_function && println("  final kernel: $key (offset=$j)")
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                body = kernel_exprs[key]
                body = remove_constants_from_kernel(body)
                body = substitute_strided_final_stage(body, current_output, current_input, j, stride, SIZE)
                push!(ops, body)
                key = "fft$(radix)_$(stride)x$(n_g)_0!"
                show_function && println("  final kernel: $key (offset=$j)")
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                body = kernel_exprs[key]
                body = remove_constants_from_kernel(body)
                body = substitute_strided_final_stage(body, current_output, current_input, j, stride, SIZE)
                push!(ops, body)
            end
        end

        # Stockham swap

        # Stockham swap
        current_input, current_output = current_output, current_input
    end

    # 4) Final body block
    return isempty(ops) ? :(copyto!(y, x)) : Expr(:block, ops...)
end

function generate_mat_execute_function!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]
    function_body = generate_mat_execute_expr!(plan, show_function)


    # 4) Final body block
    return isempty(ops) ? :(copyto!(y, x)) : Expr(:block, ops...)
end

function generate_mat_execute_function!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]
    function_body = generate_mat_execute_expr!(plan, show_function)

    func_expr = quote
        function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @inbounds begin
                $function_body
            end
            nothing
            nothing
        end
    end

    show_function && println("Generated monolithic FFT function.\n$func_expr")
    return func_expr
end


function materialize_plan_function!(plan::RadixPlan, ::Type{T}) where {T}
    body = generate_mat_execute_expr!(plan, false)

    fexpr = quote
        function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @inbounds begin
                $body
            end
            nothing
        end
    end

    return eval(fexpr)
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
            @show f
            # warmup
            Base.invokelatest(f, y, x)

            t = @belapsed Base.invokelatest($f, $y, $x)

            if t < best_time
                best_time = t
                best_body_expr = generate_mat_execute_expr!(plan, false)  # store BODY expr for compile-time splice
            end
        catch e
            @warn "Failed to benchmark plan $(plan.operations): $e"
        end
    end

    best_body_expr === nothing && error("No valid plan found")
    return best_body_expr
end

function generate_linear_execute_function!(plan::RadixPlan, show_function::Bool, ivdep::Bool)
    return generate_mat_execute_function!(plan, show_function)
end

end
