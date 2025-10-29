# ============================================================================
# Example: Using the SIMDOp System for Automated FFT Kernel Generation
# ============================================================================

include("simd_op_system.jl")
#using .SIMDOpSystem  # Adjust if you make it a module

# ============================================================================
# Example 1: Manual DAG Construction for vfft4
# ============================================================================

function manual_vfft4_dag(T::Type=Float32)
    dag = OperationDAG()
    
    # Level 0: Initial load of 8 floats (4 complex numbers)
    load_id = add_op!(dag, SIMDOp(
        OP_LOAD, -1, [], :v,
        register_size=REG_YMM,
        working_size=4,
        level=0,
        metadata=Dict(:is_contiguous => true, :memory_offset => 0)
    ))
    
    # Level 1: Split into lo and hi halves
    split_lo_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:v], :lo,
        register_size=REG_XMM,
        working_size=2,
        level=1,
        metadata=Dict(:shuffle_pattern => [1, 2, 3, 4]),
        dependencies=[load_id]
    ))
    
    split_hi_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:v], :hi,
        register_size=REG_XMM,
        working_size=2,
        level=1,
        metadata=Dict(:shuffle_pattern => [5, 6, 7, 8]),
        dependencies=[load_id]
    ))
    
    # Level 2: First butterfly (parallel ADD and SUB)
    butterfly1_id = add_op!(dag, SIMDOp(
        OP_ADD, -1, [:lo, :hi], :add_vec,
        register_size=REG_XMM,
        working_size=2,
        level=2,
        metadata=Dict(:is_butterfly => true),
        dependencies=[split_lo_id, split_hi_id]
    ))
    
    # The SUB is implicit in the butterfly
    sub_vec_id = add_op!(dag, SIMDOp(
        OP_SUB, -1, [:lo, :hi], :sub_vec,
        register_size=REG_XMM,
        working_size=2,
        level=2,
        dependencies=[split_lo_id, split_hi_id]
    ))
    
    # Level 3: Twiddle factor application
    # Shuffle sub_vec: [t2_r, t2_i, t4_r, t4_i] -> [t2_r, t2_i, t4_i, t4_r]
    shuffle_twiddle_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:sub_vec], :sub_vec_shuffled,
        register_size=REG_XMM,
        working_size=2,
        level=3,
        metadata=Dict(:shuffle_pattern => [1, 2, 4, 3]),
        dependencies=[sub_vec_id]
    ))
    
    # Sign flip on 4th element (multiply by -1 via XOR)
    signflip_id = add_op!(dag, SIMDOp(
        OP_SIGNFLIP, -1, [:sub_vec_shuffled], :sub_vec_twiddled,
        register_size=REG_XMM,
        working_size=2,
        level=3,
        metadata=Dict(:sign_pattern => (0x00000000, 0x00000000, 0x00000000, 0x80000000)),
        dependencies=[shuffle_twiddle_id]
    ))
    
    # Level 4: Interleave for second butterfly
    interleave_t12_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:add_vec, :sub_vec_twiddled], :t12,
        register_size=REG_XMM,
        working_size=2,
        level=4,
        metadata=Dict(:shuffle_pattern => [1, 2, 5, 6]),
        dependencies=[butterfly1_id, signflip_id]
    ))
    
    interleave_t34_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:add_vec, :sub_vec_twiddled], :t34,
        register_size=REG_XMM,
        working_size=2,
        level=4,
        metadata=Dict(:shuffle_pattern => [3, 4, 7, 8]),
        dependencies=[butterfly1_id, signflip_id]
    ))
    
    # Level 5: Second butterfly
    butterfly2_add_id = add_op!(dag, SIMDOp(
        OP_ADD, -1, [:t12, :t34], :y12,
        register_size=REG_XMM,
        working_size=2,
        level=5,
        dependencies=[interleave_t12_id, interleave_t34_id]
    ))
    
    butterfly2_sub_id = add_op!(dag, SIMDOp(
        OP_SUB, -1, [:t12, :t34], :y34,
        register_size=REG_XMM,
        working_size=2,
        level=5,
        dependencies=[interleave_t12_id, interleave_t34_id]
    ))
    
    # Level 6: Combine results into YMM register
    combine_id = add_op!(dag, SIMDOp(
        OP_SHUFFLE, -1, [:y12, :y34], :OUT,
        register_size=REG_YMM,
        working_size=4,
        level=6,
        metadata=Dict(:shuffle_pattern => [1, 2, 3, 4, 5, 6, 7, 8]),
        dependencies=[butterfly2_add_id, butterfly2_sub_id]
    ))
    
    # Level 7: Store result
    store_id = add_op!(dag, SIMDOp(
        OP_STORE, -1, [:OUT], :py,
        register_size=REG_YMM,
        working_size=4,
        level=7,
        dependencies=[combine_id]
    ))
    
    return dag
end

# ============================================================================
# Example 2: Automatic Saturation
# ============================================================================

"""
Demonstrate saturation of operations at different levels.
"""
function demonstrate_saturation()
    println("=" ^ 80)
    println("Demonstrating Operation Saturation")
    println("=" ^ 80)
    
    dag = manual_vfft4_dag(Float32)
    
    println("\nOriginal DAG:")
    print_dag(dag)
    
    # Detect patterns at each level
    for level in 0:7
        patterns = detect_patterns(dag, level)
        if !isempty(patterns)
            println("\nLevel $level patterns found:")
            for (i, pattern) in enumerate(patterns)
                println("  Pattern $i: Operations $(pattern)")
            end
        end
    end
    
    # Try to saturate butterfly operations
    level2_ops = filter(op -> op.level == 2, dag.ops)
    if can_saturate(level2_ops, REG_YMM)
        println("\n✓ Can saturate level 2 operations into YMM register")
        fused = saturate_butterfly(level2_ops, REG_YMM)
        println("  Fused into $(length(fused)) operation(s)")
    end
end

# ============================================================================
# Example 3: Integration with Existing recfft2_simd
# ============================================================================

"""
Modified version of recfft2_simd that builds a DAG instead of generating strings.
"""
function recfft2_to_dag_integrated(n::Int, x::Vector{String}, T::Type;
                                   level::Int=0, tmp_base::Int=1)
    dag = OperationDAG()
    
    if n == 1
        # Identity - no operations needed
        return dag, x
        
    elseif n == 2
        # Base case: create ADD and SUB operations
        
        # Parse input variable names to get indices
        x1_sym = Symbol(x[1])
        x2_sym = Symbol(x[2])
        
        # Create temporary outputs
        y1 = Symbol("t", tmp_base)
        y2 = Symbol("t", tmp_base + 1)
        
        # ADD operation
        add_id = add_op!(dag, SIMDOp(
            OP_ADD, -1, [x1_sym, x2_sym], y1,
            register_size=REG_XMM,
            working_size=1,
            level=level
        ))
        
        # SUB operation
        sub_id = add_op!(dag, SIMDOp(
            OP_SUB, -1, [x1_sym, x2_sym], y2,
            register_size=REG_XMM,
            working_size=1,
            level=level
        ))
        
        return dag, [String(y1), String(y2)]
        
    else
        # Recursive case
        n2 = n ÷ 2
        
        # Process even elements
        dag_even, t_even = recfft2_to_dag_integrated(
            n2, x[1:2:n], T;
            level=level+1, tmp_base=tmp_base
        )
        
        # Process odd elements
        dag_odd, t_odd = recfft2_to_dag_integrated(
            n2, x[2:2:n], T;
            level=level+1, tmp_base=tmp_base + 2*n2
        )
        
        # Merge DAGs
        for op in dag_even.ops
            add_op!(dag, op)
        end
        for op in dag_odd.ops
            add_op!(dag, op)
        end
        
        # Combine with butterfly
        y = String[]
        for i in 1:n2
            y_plus = "y$(i)"
            y_minus = "y$(i+n2)"
            
            # Add twiddle factor operations
            twiddle = cispi(-2*(i-1)/n)
            
            # For now, create simple operations
            # TODO: Add twiddle factor handling
            add_id = add_op!(dag, SIMDOp(
                OP_ADD, -1,
                [Symbol(t_even[i]), Symbol(t_odd[i])],
                Symbol(y_plus),
                register_size=REG_XMM,
                working_size=1,
                level=level,
                metadata=Dict(:twiddle => twiddle)
            ))
            
            sub_id = add_op!(dag, SIMDOp(
                OP_SUB, -1,
                [Symbol(t_even[i]), Symbol(t_odd[i])],
                Symbol(y_minus),
                register_size=REG_XMM,
                working_size=1,
                level=level,
                metadata=Dict(:twiddle => twiddle)
            ))
            
            push!(y, y_plus)
        end
        for i in 1:n2
            push!(y, "y$(i+n2)")
        end
        
        return dag, y
    end
end

# ============================================================================
# Example 4: Generate Optimized Kernel
# ============================================================================

"""
Complete pipeline: DAG construction -> saturation -> code generation
"""
function generate_optimized_vfft(n::Int, T::Type=Float32; kernel_name="vfft$n")
    println("=" ^ 80)
    println("Generating optimized FFT kernel for n=$n")
    println("=" ^ 80)
    
    # Step 1: Build initial DAG
    x_vars = ["x$i" for i in 1:n]
    dag, outputs = recfft2_to_dag_integrated(n, x_vars, T)
    
    println("\nStep 1: Initial DAG built")
    println("  Total operations: $(length(dag.ops))")
    
    # Step 2: Analyze levels
    levels = unique(op.level for op in dag.ops)
    println("\nStep 2: Recursion levels found: $(sort(collect(levels)))")
    
    for level in sort(collect(levels))
        level_ops = filter(op -> op.level == level, dag.ops)
        println("  Level $level: $(length(level_ops)) operations")
        
        # Count by type
        op_counts = Dict{OpType, Int}()
        for op in level_ops
            op_counts[op.op_type] = get(op_counts, op.op_type, 0) + 1
        end
        for (op_type, count) in op_counts
            println("    - $op_type: $count")
        end
    end
    
    # Step 3: Apply saturation at appropriate levels
    println("\nStep 3: Applying saturation...")
    
    # Determine saturation level based on n
    target_reg_size = if n <= 4
        REG_XMM
    elseif n <= 8
        REG_YMM
    else
        REG_YMM  # May need multiple YMM
    end
    
    println("  Target register size: $target_reg_size")
    
    # Find level where we have enough operations to saturate
    saturation_level = -1
    for level in sort(collect(levels), rev=true)
        level_ops = filter(op -> op.level == level, dag.ops)
        if can_saturate(level_ops, target_reg_size)
            saturation_level = level
            break
        end
    end
    
    if saturation_level >= 0
        println("  Saturation level: $saturation_level")
        # TODO: Actually perform saturation
    else
        println("  No saturation possible with current register size")
    end
    
    # Step 4: Generate code
    println("\nStep 4: Generating Julia code...")
    code = generate_kernel_code(dag, T, kernel_name)
    
    return code, dag
end

# ============================================================================
# Run Examples
# ============================================================================

function run_examples()
    println("\n" * "=" ^ 80)
    println("SIMD FFT Kernel Generator Examples")
    println("=" ^ 80)
    
    # Example 1: Manual DAG
    println("\n--- Example 1: Manual vfft4 DAG ---")
    dag = manual_vfft4_dag(Float32)
    print_dag(dag)
    
    # Example 2: Saturation demonstration
    println("\n--- Example 2: Operation Saturation ---")
    demonstrate_saturation()
    
    # Example 3: Generate vfft4
    println("\n--- Example 3: Generate vfft4 ---")
    code, dag = generate_optimized_vfft(4, Float32)
    println("\nGenerated code:")
    println(code)
    
    # Example 4: Generate vfft8
    println("\n--- Example 4: Generate vfft8 ---")
    code8, dag8 = generate_optimized_vfft(8, Float32)
    println("\nGenerated code:")
    println(code8)
end

# Uncomment to run:
run_examples()

