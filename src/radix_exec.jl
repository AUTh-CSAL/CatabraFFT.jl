module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools
using SIMD

include("helper_tools.jl")

# Generate a complete monolithic FFT function with all kernels inlined
function GenerateMatrixExpr!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]

    # Gather all kernel expressions
    plan_data = (n=plan.n, operations=plan.operations)
    kernel_exprs = RadixGenerator.create_kernel_dictionary(plan_data, T)

    show_function && println("Available kernels: ", collect(keys(kernel_exprs)))

    # Collect constants once at the top
    ops = Expr[]
    constants_dict = extract_constants_dict(kernel_exprs)
    sizehint!(ops, length(constants_dict) + length(plan.operations) + 5)
    
    @inbounds for (const_name, const_value) in constants_dict
        push!(ops, :($(Symbol(const_name)) = $const_value))
    end

    @inbounds for (stage_idx, op) in enumerate(plan.operations)
        current_input  = op.eo ? :y : :x
        current_output = op.eo ? :x : :y

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
            push!(ops, kernel_exprs[key])
            
        elseif !is_final_stage 
            n_groups_per_radix = SIZE ÷ radix
            
            for p in 0:(n_groups_per_radix-1)
                key = "fft$(radix)_$(stride)x$(n_g)_$(p)!"
                haskey(kernel_exprs, key) || error("Missing kernel: $key")
                show_function && println("  Sub-kernel: $key")
                push!(ops, kernel_exprs[key])
            end
        else
            # Terminal stage: loop-based execution
            key = "fft$(radix)_$(stride)x$(n_g)_0!"
            haskey(kernel_exprs, key) || error("Missing kernel: $key")
            
            show_function && println("  Terminal kernel: $key (looped $stride times)")
            
            # Determine correct output based on stage parity
            current_output = (length(plan.operations) % 2 == 0) ? :x : :y

            loop_body = substitute_strided_final_loop(kernel_exprs[key], current_output, current_input, stride, SIZE, radix)
            
            # Wrap in vectorized loop
            loop_expr = quote
                @inbounds @simd ivdep for idx in 1:$stride
                    $loop_body
                end
            end
            
            push!(ops, loop_expr)
        end
    end

    return Expr(:block, ops...)
end

function generate_mat_execute_function!(plan::RadixPlan, show_function::Bool=false)::Expr
    T = typeof(plan).parameters[1]
    function_body = GenerateMatrixExpr!(plan, show_function)
    
    show_function && println("Generated function body")

    func_expr = quote
        @inline function (y::AbstractVector{Complex{$T}}, x::AbstractVector{Complex{$T}})
            @fastmath @inbounds begin
                $function_body
            end
            nothing
        end
    end

    show_function && println("Generated monolithic FFT function")
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
    
    return eval(fexpr)
end

# Efficiently substitute constant symbols with literal values
function substitute_constants_in_expr(expr, constants_dict::Dict{Symbol, T}) where T
    if isa(expr, Expr)
        # Recursively process all sub-expressions
        return Expr(expr.head, [substitute_constants_in_expr(arg, constants_dict) for arg in expr.args]...)
    elseif isa(expr, Symbol)
        # Replace symbol with literal value if it's a constant
        return get(constants_dict, expr, expr)
    else
        return expr
    end
end

# Extract constants as dictionary to avoid duplicates
function extract_constants_dict(kernel_exprs::Dict{String, Expr})
    constants_dict = Dict{String, Any}()
    
    @inbounds for (name, expr) in kernel_exprs
        extract_constants_recursive!(constants_dict, expr)
    end
    
    return constants_dict
end

function extract_constants_recursive!(constants::Dict{String, Any}, expr)
    if isa(expr, Expr)
        if expr.head == :(=) && length(expr.args) == 2
            lhs = expr.args[1]
            if isa(lhs, Symbol)
                lhs_str = string(lhs)
                # Check if it's a constant definition
                if lhs_str == "INV_SQRT2" || startswith(lhs_str, "COSPI_") || startswith(lhs_str, "SINPI_")
                    if !haskey(constants, lhs_str)
                        constants[lhs_str] = expr.args[2]
                    end
                end
            end
        else
            # Recursively process all arguments
            @inbounds for arg in expr.args
                if isa(arg, Expr)
                    extract_constants_recursive!(constants, arg)
                end
            end
        end
    end
end

function substitute_strided_final_loop(kernel_expr::Expr, out_var, in_var, stride::Int, size::Int, radix::Int)
    out_sym = out_var isa Symbol ? out_var : Symbol(out_var)
    in_sym = in_var isa Symbol ? in_var : Symbol(in_var)
    
    # Recursive transformation with tail-call optimization hint
    @inline function transform(ex, is_lhs::Bool)
        if isa(ex, Expr)
            if ex.head == :ref && length(ex.args) == 2
                arr_sym = ex.args[1] isa Symbol ? ex.args[1] : Symbol(ex.args[1])
                idx = ex.args[2]
                
                # Transform array[k] → array[k + idx - 1]
                if (arr_sym == out_sym || arr_sym == in_sym) && isa(idx, Int)
                    offset = idx - 1
                    return offset == 0 ? Expr(:ref, arr_sym, :idx) : Expr(:ref, arr_sym, :(idx + $offset))
                end
                
            elseif ex.head == :(=)
                return Expr(:(=), transform(ex.args[1], true), transform(ex.args[2], false))
            else
                return Expr(ex.head, [transform(arg, is_lhs) for arg in ex.args]...)
            end
        end
        return ex
    end
    
    return transform(kernel_expr, false)
end

# Benchmarking function with optimized execution
function return_best_static_linear_expr(plans::Vector{RadixPlan{T}}, show_function::Bool)::Expr where T<:AbstractFloat
    @assert !isempty(plans)
    N = plans[1].n

    # Fixed inputs for fair timing
    x = rand(Complex{T}, N)
    y = similar(x)

    best_time = Inf
    best_body_expr::Union{Expr,Nothing} = nothing
    
    @inbounds for plan in plans
        try
            show_function && println("Benchmarking plan: ", plan.operations)

            f = materialize_plan_function!(plan, T)
            
            show_function && println("Materialized function")
            
            # Warmup
            Base.invokelatest(f, y, x)

            # Benchmark
            t = @belapsed Base.invokelatest($f, $y, $x)
            
            show_function && println("Benchmarked time: $t")

            if t < best_time
                best_time = t
                best_body_expr = GenerateMatrixExpr!(plan, show_function)
            end
        catch e
            @warn "Failed to benchmark plan $(plan.operations): $e"
        end
    end
    
    show_function && println("Best time: $best_time")

    best_body_expr === nothing && error("No valid plan found")
    return best_body_expr
end

function generate_linear_execute_function!(plan::RadixPlan, show_function::Bool, ivdep::Bool)
    return GenerateMatrixExpr!(plan, show_function)
end

end