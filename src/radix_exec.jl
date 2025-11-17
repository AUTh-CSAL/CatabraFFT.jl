module Radix_Execute

using Core.Compiler: Core, return_type
using ..Radix_Plan
using ..RadixGenerator
using BenchmarkTools
using SIMD

include("helper_tools.jl")

#TODO : Experiment with possible permutation of specific instructions across different layers. 
# Possible cache friendly patterns to be utilized. Fully saturate a part of the next layer before the previous layer is done computing. 
# ex. fft4xfft2. Isn't this just split-radix ???? Automate this for other pairs (8-4).
# What about AVX vs VSML usage???

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
        current_input, current_output = op.input_buffer, op.output_buffer

        is_final_stage = (stage_idx == length(plan.operations))
        radix = get_radix_divisor(op.op_type)
        n_g = op.n_groups
        stride = op.stride
        SIZE = n_g * stride
        is_monolithic_shell = (radix == n_g) && (stride == 1)
        
        show_function && println("Stage $stage_idx: radix=$radix, n_groups=$n_g, stride=$stride, in=$current_input, out=$current_output")

        if is_monolithic_shell
            key = "fft$(radix)_shell!"
            show_function && println("  kernel: $key")
            haskey(kernel_exprs, key) || error("Missing kernel: $key")
            push!(ops, kernel_exprs[key])
            
        elseif !is_final_stage 
            n_groups_per_radix = SIZE ÷ radix
            
            @inbounds @simd for p in 0:(n_groups_per_radix-1)
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
            #current_output = (length(plan.operations) % 2 == 0) ? :x : :y

            loop_body = substitute_strided_final_loop(kernel_exprs[key], current_output, current_input, stride, SIZE, radix)

            show_function && @show loop_body

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
        @inline function (y_complex::AbstractVector{Complex{$T}}, x_complex::AbstractVector{Complex{$T}})
            # Reinterpret Complex{T} arrays as T arrays for kernel access
            # Complex numbers are stored as consecutive pairs: [real1, imag1, real2, imag2, ...]
            x = reinterpret($T, x_complex)
            y = reinterpret($T, y_complex)

            @fastmath @inbounds begin
                $function_body
            end

            y_complex = reinterpret(Complex{$T}, y)

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
        @inline function (y_complex::AbstractVector{Complex{$T}}, x_complex::AbstractVector{Complex{$T}})
            # Reinterpret Complex{T} arrays as T arrays for kernel access
            # Complex numbers are stored as consecutive pairs: [real1, imag1, real2, imag2, ...]
            x = reinterpret($T, x_complex)
            y = reinterpret($T, y_complex)

            @fastmath @inbounds begin
                $substituted_body
            end

            y_complex = reinterpret(Complex{$T}, y)

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

    # Recursive transformation for Stockham algorithm final stage
    # For radix-R butterfly at iteration i:
    # - Inputs come from x at positions: i, i+stride, i+2*stride, ..., i+(R-1)*stride
    # - Outputs go to y at positions: i, i+stride, i+2*stride, ..., i+(R-1)*stride
    # With real/imag interleaving: position k = 2*k + [0 for real, 1 for imag]
    @inline function transform(ex, is_lhs::Bool)
        if isa(ex, Expr)
            if ex.head == :ref && length(ex.args) == 2
                arr_sym = ex.args[1] isa Symbol ? ex.args[1] : Symbol(ex.args[1])
                idx_val = ex.args[2]

                # Transform array indices for strided Stockham pattern
                if (arr_sym == out_sym || arr_sym == in_sym) && isa(idx_val, Int)
                    # Determine which complex element this is (0-indexed)
                    # Float indices come in pairs: [1,2] = complex 0, [3,4] = complex 1, etc.
                    complex_num = (idx_val - 1) ÷ 2  # Which complex element (0, 1, 2, ...)
                    is_imag = (idx_val - 1) % 2 == 1  # Is this the imaginary part?

                    # Stockham final stage: element k of iteration idx goes to position (idx + k*stride)
                    # In float indices: 2*(idx + k*stride) - 1 for real, 2*(idx + k*stride) for imag
                    # Note: idx is 1-indexed loop variable (1 to stride)
                    if complex_num == 0
                        # First complex: position idx
                        return is_imag ? Expr(:ref, arr_sym, :(2*idx)) : Expr(:ref, arr_sym, :(2*idx - 1))
                    else
                        # Other complex: position idx + k*stride
                        offset = 2 * complex_num * stride
                        return is_imag ?
                            Expr(:ref, arr_sym, :(2*idx + $offset)) :
                            Expr(:ref, arr_sym, :(2*idx + $(offset - 1)))
                    end
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
# Returns a NamedTuple with:
#   - expr: the best body expression
#   - benchmarks: dictionary mapping plan operations to benchmark results
function return_best_static_linear_expr(plans::Vector{RadixPlan{T}}, show_function::Bool) where T<:AbstractFloat
    @assert !isempty(plans)
    N = plans[1].n

    # Fixed inputs for fair timing
    x = rand(Complex{T}, N)
    y = similar(x)

    best_time = Inf
    best_body_expr::Union{Expr,Nothing} = nothing
    best_plan_idx = 0

    # Dictionary to store all benchmark results
    # Key: index in plans vector
    # Value: NamedTuple with plan, time, function, and expression
    benchmark_results = Dict{Int, NamedTuple}()

    @inbounds for (idx, plan) in enumerate(plans)
        try
            show_function && println("Benchmarking plan: ", plan.operations)

            f = materialize_plan_function!(plan, T)

            show_function && println("Materialized function")

            # Warmup
            Base.invokelatest(f, y, x)

            # Benchmark
            t = @belapsed Base.invokelatest($f, $y, $x)

            show_function && println("Benchmarked time: $t")

            # Generate body expression
            body_expr = GenerateMatrixExpr!(plan, show_function)

            # Store benchmark result
            benchmark_results[idx] = (
                plan = plan,
                time = t,
            )

            if t < best_time
                best_time = t
                best_body_expr = body_expr
                best_plan_idx = idx
            end
        catch e
            @warn "Failed to benchmark plan $(plan.operations): $e"
        end
    end

    show_function && println("Best time: $best_time (plan #$best_plan_idx)")

    best_body_expr === nothing && error("No valid plan found")

    return (expr = best_body_expr, benchmarks = benchmark_results)
end

function generate_linear_execute_function!(plan::RadixPlan, show_function::Bool, ivdep::Bool)
    return GenerateMatrixExpr!(plan, show_function)
end

end
