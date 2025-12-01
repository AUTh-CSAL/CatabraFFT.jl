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

    # Recursive transformation for a generic strided Stockham stage.
    # This function transforms contiguous template indices (e.g., y[1], y[17])
    # into strided loop indices (e.g., y[2*idx - 1], y[2*idx - 1 + 32])
    # based on the decomposition geometry (size, radix, stride).
    @inline function transform(ex, is_lhs::Bool)
        if isa(ex, Expr)
            if ex.head == :ref && length(ex.args) == 2
                arr_sym = ex.args[1] isa Symbol ? ex.args[1] : Symbol(ex.args[1])
                idx_val = ex.args[2]

                if (arr_sym == out_sym || arr_sym == in_sym) && isa(idx_val, Int)
                    
                    # 1. Determine the complex index (0-based) in the template.
                    # e.g., index 1 (real) -> (1-1)//2 = 0, index 2 (imag) -> (2-1)//2 = 0
                    template_complex_num = (idx_val - 1) ÷ 2 
                    is_imag = (idx_val - 1) % 2 == 1  # Is this the imaginary part?

                    k::Int = 0 # k is the complex offset multiplier
                    
                    if is_lhs # STORE indices (Outputs of the butterfly)
                        # Store pattern for DIF Stockham is contiguous in the output block.
                        # The complex offset scales directly with the current stage stride.
                        # k is simply the complex index (0, 1, 2, 3...)
                        k = template_complex_num 
                        
                    else # LOAD indices (Inputs of the butterfly)
                        # Load pattern for DIF Stockham is strided.
                        # The template kernel was generated for a total block size of 'size'.
                        # The inputs to a butterfly of size 'size' with radix 'radix'
                        # are separated by 'size ÷ radix'. This is the implicit stride 
                        # baked into the template's indices.
                        
                        implicit_template_load_stride::Int = size ÷ radix
                        
                        if implicit_template_load_stride == 0
                            # Should not happen for valid FFTs (size >= radix)
                            implicit_template_load_stride = 1 
                        end

                        # k is the complex offset multiplier determined by the striding 
                        # within the template kernel indices.
                        # e.g. for N=32, radix=2, inputs are at 0 and 16. 
                        # implicit_stride = 16. k values are 0/16=0 and 16/16=1.
                        k = template_complex_num ÷ implicit_template_load_stride
                    end
                    
                    # 2. The float offset (MUST be a compile-time constant)
                    # Float Offset = k (complex offset) * stride (stage stride) * 2 (floats per complex)
                    strided_float_offset::Int = k * stride * 2

                    # 3. Construct the clean indexing expression.
                    # The total constant offset C is the strided offset plus 1 if imaginary.
                    total_const_offset::Int = strided_float_offset + (is_imag ? 1 : 0)

                    # The base variable part of the index: (2 * idx - 1)
                    # This assumes the loop variable 'idx' is 1-based and iterates over complex pairs.
                    base_var_expr = :((2 * idx) - 1)
                    
                    if total_const_offset == 0
                        # Simplest case: y[2idx - 1]. No redundant + 0.
                        index_expr = base_var_expr
                    else
                        # General case: y[(2idx - 1) + C]. We use a saturated quote to insert the constant.
                        index_expr = :($base_var_expr + $total_const_offset)
                    end
                    
                    return Expr(:ref, arr_sym, index_expr)
                end

            # Handle vload/vstore calls: vload(Vec{N,T}, array, position) or vstore(value, array, position)
            elseif ex.head == :call && length(ex.args) >= 3
                func_name = ex.args[1]
                if func_name == :vload && length(ex.args) == 4
                    # vload(Vec{N,T}, array_sym, position)
                    vec_type = ex.args[2]
                    arr_arg = ex.args[3]
                    pos_arg = ex.args[4]
                    arr_sym_check = arr_arg isa Symbol ? arr_arg : Symbol(arr_arg)

                    if (arr_sym_check == out_sym || arr_sym_check == in_sym) && isa(pos_arg, Int)
                        # Transform position using same logic as array references
                        idx_val = pos_arg
                        template_complex_num = (idx_val - 1) ÷ 2
                        is_imag = (idx_val - 1) % 2 == 1

                        # vload is always a load (is_lhs=false)
                        implicit_template_load_stride = size ÷ radix
                        if implicit_template_load_stride == 0
                            implicit_template_load_stride = 1
                        end
                        k = template_complex_num ÷ implicit_template_load_stride

                        strided_float_offset = k * stride * 2
                        total_const_offset = strided_float_offset + (is_imag ? 1 : 0)
                        base_var_expr = :((2 * idx) - 1)

                        pos_expr = if total_const_offset == 0
                            base_var_expr
                        else
                            :($base_var_expr + $total_const_offset)
                        end

                        return Expr(:call, :vload, vec_type, arr_arg, pos_expr)
                    end

                elseif func_name == :vstore && length(ex.args) == 4
                    # vstore(value, array_sym, position)
                    value_arg = ex.args[2]
                    arr_arg = ex.args[3]
                    pos_arg = ex.args[4]
                    arr_sym_check = arr_arg isa Symbol ? arr_arg : Symbol(arr_arg)

                    if (arr_sym_check == out_sym || arr_sym_check == in_sym) && isa(pos_arg, Int)
                        # Transform position using same logic as array references
                        idx_val = pos_arg
                        template_complex_num = (idx_val - 1) ÷ 2
                        is_imag = (idx_val - 1) % 2 == 1

                        # vstore is always a store (is_lhs=true)
                        k = template_complex_num

                        strided_float_offset = k * stride * 2
                        total_const_offset = strided_float_offset + (is_imag ? 1 : 0)
                        base_var_expr = :((2 * idx) - 1)

                        pos_expr = if total_const_offset == 0
                            base_var_expr
                        else
                            :($base_var_expr + $total_const_offset)
                        end

                        # Transform value recursively (it might contain loads)
                        transformed_value = transform(value_arg, false)
                        return Expr(:call, :vstore, transformed_value, arr_arg, pos_expr)
                    end
                end
                # Fall through to general case if not matched

            # Recursive traversal for assignment expressions
            elseif ex.head == :(=)
                # Apply the strided logic to both LHS (Store - is_lhs=true) and RHS (Load - is_lhs=false)
                return Expr(:(=), transform(ex.args[1], true), transform(ex.args[2], false))

            # Recursive traversal for tuples, blocks, calls, etc.
            else
                # Clean up metadata lines from the initial quote (like #= none:3 =#)
                new_args = []
                for arg in ex.args
                    if isa(arg, Expr) && arg.head == :line
                        continue
                    end
                    push!(new_args, transform(arg, is_lhs))
                end
                return Expr(ex.head, new_args...)
            end
        end
        return ex
    end

    # The transform function is called once on the entire kernel expression
    kernel_expr = transform(kernel_expr, false)
    
    return kernel_expr
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
