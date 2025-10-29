# Integration Guide: Connecting SIMDOp System with Your Existing Code

## Current State Analysis

Your `fft_seed.jl` has:
- `recfft2_simd()`: Recursive FFT generator (string-based)
- `load_real_imag_gen()`: Generates load code
- `sat_expr_simd_scalar()`: Twiddle factor application
- `generate_butterfly_combination()`: Butterfly merge code

## Integration Strategy

### Phase 1: Hybrid Approach (Easiest)

Keep existing code but add operation tracking:

```julia
include("simd_op_system.jl")

# Global DAG that accumulates operations
global_dag = nothing

function recfft2_simd_tracked(y, x, d, w, root, ::Type{T}, tmp_base, mode, py, complexes_per_vec;
                               dag=nothing, level=0) where T
    # Create DAG if this is the root call
    if dag === nothing
        global global_dag = OperationDAG()
        dag = global_dag
    end
    
    # Your existing logic...
    if n == 1
        # Track NOP
        add_op!(dag, SIMDOp(OP_NOP, -1, [], Symbol(y[1]), level=level))
        # Return existing string code
        return ""
        
    elseif n == 2
        # Track operations while generating strings
        x1_sym = Symbol(replace(x[1], r"[^a-zA-Z0-9_]" => "_"))
        x2_sym = Symbol(replace(x[2], r"[^a-zA-Z0-9_]" => "_"))
        y1_sym = Symbol(y[1])
        y2_sym = Symbol(y[2])
        
        # Track ADD
        add_op!(dag, SIMDOp(
            OP_ADD, -1, [x1_sym, x2_sym], y1_sym,
            register_size=REG_XMM, working_size=1, level=level
        ))
        
        # Track SUB
        add_op!(dag, SIMDOp(
            OP_SUB, -1, [x1_sym, x2_sym], y2_sym,
            register_size=REG_XMM, working_size=1, level=level
        ))
        
        # Return existing string code
        if root
            return """
            x1_r, x1_i = real($(x[1])), imag($(x[1]))
            x2_r, x2_i = real($(x[2])), imag($(x[2]))
            $(y[1]) = Complex{$T}(x1_r + x2_r, x1_i + x2_i)
            """
        else
            return """
            $(y[1])_r, $(y[1])_i = $(x[1])_r + $(x[2])_r, $(x[1])_i + $(x[2])_i
            $(y[2])_r, $(y[2])_i = $(x[1])_r - $(x[2])_r, $(x[1])_i - $(x[2])_i
            """
        end
        
    else
        # Recursive case - track and recurse
        n2 = n ÷ 2
        
        # ... existing recursive logic ...
        s1 = recfft2_simd_tracked(t[1:vecs_per_half], x[1:2:n], nothing, nothing, 
                                  false, T, new_tmp_base, mode, py, complexes_per_vec;
                                  dag=dag, level=level+1)
        
        s2 = recfft2_simd_tracked(t[vecs_per_half+1:2*vecs_per_half], x[2:2:n], nothing,
                                  get_twiddle_expression(collect(0:n2-1), n; T=T, accuracy=nothing),
                                  false, T, new_tmp_base, mode, py, complexes_per_vec;
                                  dag=dag, level=level+1)
        
        # Track butterfly combination
        for i in 1:n2
            t_even_sym = Symbol(t[i])
            t_odd_sym = Symbol(t[i+n2])
            y_plus_sym = Symbol(y[i])
            y_minus_sym = Symbol(y[n2+i])
            
            # Track ADD butterfly
            add_op!(dag, SIMDOp(
                OP_ADD, -1, [t_even_sym, t_odd_sym], y_plus_sym,
                register_size=REG_XMM, working_size=1, level=level,
                metadata=Dict(:is_butterfly => true)
            ))
            
            # Track SUB butterfly  
            add_op!(dag, SIMDOp(
                OP_SUB, -1, [t_even_sym, t_odd_sym], y_minus_sym,
                register_size=REG_XMM, working_size=1, level=level,
                metadata=Dict(:is_butterfly => true)
            ))
        end
        
        s3p, s3m = generate_butterfly_combination(y, t, d, w, n, n2, root, T, py)
        
        return s1 * s2 * s3p * s3m
    end
end

# Usage:
code = recfft2_simd_tracked(["y$i" for i in 1:4], ["x$i" for i in 1:4],
                            nothing, nothing, true, Float32, 1, :vgather, "py", 4)

# Now you have both:
# 1. The string code (for comparison/validation)
# 2. The DAG (for analysis and optimization)

println("Generated code:")
println(code)

println("\nDAG structure:")
print_dag(global_dag)

# Analyze saturation opportunities
for level in 0:3
    level_ops = filter(op -> op.level == level, global_dag.ops)
    println("\nLevel $level: $(length(level_ops)) operations")
    if can_saturate(level_ops, REG_YMM)
        println("  ✓ Can saturate into YMM")
    end
end
```

### Phase 2: Pure DAG Approach (More Work, Better Results)

Replace string generation entirely:

```julia
function recfft2_dag_only(n::Int, x_indices::Vector{Int}, T::Type;
                          level=0, tmp_base=Ref(1), with_twiddle=false)
    dag = OperationDAG()
    
    if n == 1
        # Identity
        return dag, [x_indices[1]]
        
    elseif n == 2
        # Create symbols for inputs
        x1 = Symbol("x", x_indices[1])
        x2 = Symbol("x", x_indices[2])
        
        # Output temporaries
        t1 = Symbol("t", tmp_base[])
        t2 = Symbol("t", tmp_base[] + 1)
        tmp_base[] += 2
        
        # ADD operation
        add_op!(dag, SIMDOp(
            OP_ADD, -1, [x1, x2], t1,
            register_size=REG_XMM, working_size=1, level=level
        ))
        
        # SUB operation with optional twiddle
        if with_twiddle
            # Will apply twiddle factor later
            add_op!(dag, SIMDOp(
                OP_SUB, -1, [x1, x2], t2,
                register_size=REG_XMM, working_size=1, level=level,
                metadata=Dict(:needs_twiddle => true)
            ))
        else
            add_op!(dag, SIMDOp(
                OP_SUB, -1, [x1, x2], t2,
                register_size=REG_XMM, working_size=1, level=level
            ))
        end
        
        return dag, [tmp_base[] - 2, tmp_base[] - 1]
        
    else
        n2 = n ÷ 2
        
        # Even recursion
        dag_even, t_even_indices = recfft2_dag_only(
            n2, x_indices[1:2:n], T;
            level=level+1, tmp_base=tmp_base, with_twiddle=false
        )
        
        # Odd recursion with twiddle
        dag_odd, t_odd_indices = recfft2_dag_only(
            n2, x_indices[2:2:n], T;
            level=level+1, tmp_base=tmp_base, with_twiddle=true
        )
        
        # Merge DAGs
        for op in dag_even.ops
            add_op!(dag, op)
        end
        for op in dag_odd.ops
            add_op!(dag, op)
        end
        
        # Apply twiddles to odd branch
        for (i, t_idx) in enumerate(t_odd_indices)
            if i > 1  # First element doesn't need twiddle
                t_sym = Symbol("t", t_idx)
                t_twiddled = Symbol("t", tmp_base[])
                tmp_base[] += 1
                
                twiddle_factor = cispi(-2*(i-1)/n)
                
                add_op!(dag, SIMDOp(
                    OP_TWIDDLE, -1, [t_sym], t_twiddled,
                    register_size=REG_XMM, working_size=1, level=level,
                    metadata=Dict(:twiddle => twiddle_factor)
                ))
                
                t_odd_indices[i] = tmp_base[] - 1
            end
        end
        
        # Butterfly combination
        y_indices = Int[]
        for i in 1:n2
            t_even = Symbol("t", t_even_indices[i])
            t_odd = Symbol("t", t_odd_indices[i])
            
            y_plus = Symbol("y", tmp_base[])
            y_minus = Symbol("y", tmp_base[] + 1)
            tmp_base[] += 2
            
            # ADD
            add_op!(dag, SIMDOp(
                OP_ADD, -1, [t_even, t_odd], y_plus,
                register_size=REG_XMM, working_size=1, level=level,
                metadata=Dict(:is_butterfly => true)
            ))
            
            # SUB
            add_op!(dag, SIMDOp(
                OP_SUB, -1, [t_even, t_odd], y_minus,
                register_size=REG_XMM, working_size=1, level=level,
                metadata=Dict(:is_butterfly => true)
            ))
            
            push!(y_indices, tmp_base[] - 2)
            push!(y_indices, tmp_base[] - 1)
        end
        
        return dag, y_indices
    end
end

# Usage:
dag, output_indices = recfft2_dag_only(4, [1, 2, 3, 4], Float32)
print_dag(dag)

# Now apply saturation
saturated_dag = apply_saturation(dag, REG_YMM)

# Generate code
code = generate_kernel_code(saturated_dag, Float32, "vfft4_auto")
println(code)
```

### Phase 3: Complete Automation

Create a single entry point:

```julia
"""
Generate optimized SIMD FFT kernel of size n.

# Arguments
- `n::Int`: FFT size (must be power of 2)
- `T::Type`: Element type (Float32 or Float64)
- `target::Symbol`: Target ISA (:avx2, :avx512, :neon)
- `kernel_name::String`: Name of generated function

# Returns
- Generated Julia code as string
"""
function generate_fft_kernel(n::Int, T::Type=Float32;
                             target::Symbol=:avx2,
                             kernel_name="vfft$n")
    @assert ispow2(n) "n must be power of 2"
    
    # Determine target register size
    reg_size = if target == :avx2
        REG_YMM
    elseif target == :avx512
        REG_ZMM
    else
        REG_XMM
    end
    
    # Build DAG
    println("Building computation DAG for n=$n...")
    dag, _ = recfft2_dag_only(n, collect(1:n), T)
    println("  $(length(dag.ops)) operations created")
    
    # Analyze levels
    max_level = maximum(op.level for op in dag.ops)
    println("\nRecursion depth: $max_level levels")
    
    for level in 0:max_level
        level_ops = filter(op -> op.level == level, dag.ops)
        println("  Level $level: $(length(level_ops)) operations")
    end
    
    # Apply saturation
    println("\nApplying saturation to target=$target...")
    saturated_dag = apply_saturation(dag, reg_size)
    println("  Reduced to $(length(saturated_dag.ops)) operations")
    
    # Generate code
    println("\nGenerating Julia code...")
    code = generate_kernel_code(saturated_dag, T, kernel_name)
    
    return code
end

# Generate family of kernels
for n in [2, 4, 8, 16, 32]
    code = generate_fft_kernel(n, Float32, target=:avx2)
    
    # Save to file
    open("generated_vfft$(n).jl", "w") do f
        write(f, code)
    end
    
    println("Generated vfft$(n)")
end
```

## Validation Strategy

Compare generated code against hand-written:

```julia
using Test, FFTW

function validate_kernel(n::Int, T::Type)
    # Generate kernel
    code = generate_fft_kernel(n, T)
    eval(Meta.parse(code))
    kernel_func = eval(Symbol("vfft$n"))
    
    # Test against FFTW
    for trial in 1:100
        x_complex = rand(ComplexF32, n)
        x_float = reinterpret(Float32, x_complex)
        y_float = similar(x_float)
        
        # Run generated kernel
        kernel_func(x_float, y_float)
        y_complex = reinterpret(ComplexF32, y_float)
        
        # Compare with FFTW
        expected = fft(x_complex)
        
        @test isapprox(y_complex, expected, rtol=1e-5)
    end
    
    println("✓ vfft$n validated against FFTW")
end

# Validate all generated kernels
for n in [2, 4, 8, 16]
    validate_kernel(n, Float32)
end
```

## Performance Comparison

```julia
using BenchmarkTools

function benchmark_kernels(n::Int)
    # Setup
    x = rand(Float32, 2*n)
    y = similar(x)
    
    # Hand-written (if exists)
    if isdefined(Main, Symbol("vfft$(n)_manual"))
        manual_time = @belapsed $(Symbol("vfft$(n)_manual"))($x, $y)
        println("Manual vfft$n: $(manual_time*1e9) ns")
    end
    
    # Generated
    auto_time = @belapsed $(Symbol("vfft$n"))($x, $y)
    println("Generated vfft$n: $(auto_time*1e9) ns")
    
    # FFTW
    x_complex = reinterpret(ComplexF32, x)
    fftw_time = @belapsed fft!($x_complex)
    println("FFTW: $(fftw_time*1e9) ns")
    
    println()
end

for n in [4, 8, 16]
    println("n=$n:")
    benchmark_kernels(n)
end
```

## Summary

1. **Phase 1 (Hybrid)**: Track operations while keeping string generation
   - Lowest risk, immediate insights
   - Can analyze DAG structure
   - Validate correctness

2. **Phase 2 (Pure DAG)**: Replace string generation with DAG building
   - Better optimization opportunities
   - Cleaner code
   - More work to implement

3. **Phase 3 (Automation)**: Single function generates any kernel
   - Ultimate goal
   - Generates optimal code for any n
   - Easy to extend (new ISAs, optimizations)

Start with Phase 1 to understand the system, then gradually migrate to Phase 2 and 3.

