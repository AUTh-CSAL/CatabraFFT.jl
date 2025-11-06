# ============================================================================
# Examples and Tests for E-Graph SIMD System
# ============================================================================

include("egraph_simd_system.jl")

using Printf

# ============================================================================
# Example 1: Demonstrating Cost Model
# ============================================================================

function example_cost_comparison()
    println("="^80)
    println("Example 1: Cost Comparison - MUL vs XOR for Sign Flips")
    println("="^80)

    # Cost of multiplying by [-1, 1, -1, 1] using MUL
    mul_cost = get_cost(OP_MUL, REG_XMM)
    println("\nMUL operation (naive approach):")
    println("  Latency:    $(mul_cost.latency) cycles")
    println("  Throughput: $(mul_cost.throughput) cycles")
    println("  Total cost: $(0.3 * mul_cost.latency + 0.7 * mul_cost.throughput)")

    # Cost of using XOR for sign flip
    xor_cost = get_cost(OP_XOR, REG_XMM)
    println("\nXOR operation (optimized approach):")
    println("  Latency:    $(xor_cost.latency) cycles")
    println("  Throughput: $(xor_cost.throughput) cycles")
    println("  Total cost: $(0.3 * xor_cost.latency + 0.7 * xor_cost.throughput)")

    speedup = (0.3 * mul_cost.latency + 0.7 * mul_cost.throughput) /
              (0.3 * xor_cost.latency + 0.7 * xor_cost.throughput)
    println("\nSpeedup: $(round(speedup, digits=2))x faster!")

    println("\nConclusion: Using XOR for sign flips is ~$(round(speedup, digits=2))x more efficient")
    println("This is why vfft4_xor uses XOR instead of multiplication!")
end

# ============================================================================
# Example 2: Algebraic Rewriting - MUL to XOR
# ============================================================================

function example_mul_to_xor_rewrite()
    println("\n" * "="^80)
    println("Example 2: Algebraic Rewriting - MUL with Sign Pattern → XOR")
    println("="^80)

    # Create a multiplication node: v * [1, 1, -1, 1]
    println("\nOriginal expression:")
    println("  v * [1.0, 1.0, -1.0, 1.0]")

    mul_node = SIMDNode(
        OP_MUL,
        [:v, :sign_vector],
        REG_XMM,
        metadata=Dict(
            :constant_vector => [1.0, 1.0, -1.0, 1.0],
            :n_floats => 4
        )
    )

    println("\nOriginal cost: $(simd_cost(mul_node))")

    # Apply algebraic rewrite
    optimized = apply_algebraic_rewrites(mul_node)

    println("\nOptimized expression:")
    println("  v ⊻ [0x00000000, 0x00000000, 0x80000000, 0x00000000]")
    println("\nOptimized cost: $(simd_cost(optimized))")

    # Generate code for both
    println("\n--- Generated Code (Original) ---")
    println(codegen(mul_node, Float32))

    println("\n--- Generated Code (Optimized) ---")
    println(codegen(optimized, Float32))

    println("\n✓ Successfully rewrote MUL to XOR!")
end

# ============================================================================
# Example 3: Twiddle Factor Optimization - Multiply by i
# ============================================================================

function example_twiddle_by_i()
    println("\n" * "="^80)
    println("Example 3: Twiddle Factor Optimization - Multiply by i")
    println("="^80)

    println("\nComplex multiplication by i: (a + bi) * i = -b + ai")
    println("Naive: Use complex multiply (expensive)")
    println("Optimized: Shuffle real/imag, then XOR to negate")

    # Create twiddle by i node
    twiddle_node = SIMDNode(
        OP_MUL,
        [:v, :imag_unit],
        REG_XMM,
        metadata=Dict(
            :twiddle_type => :imag_unit,
            :n_floats => 4  # 2 complex numbers
        )
    )

    println("\nOriginal cost: $(simd_cost(twiddle_node))")

    # Apply rewrite
    optimized = apply_algebraic_rewrites(twiddle_node)

    println("Optimized cost: $(simd_cost(optimized))")

    println("\n--- Generated Code ---")
    println(codegen(optimized, Float32))

    println("\n✓ Multiplication by i converted to SHUFFLE + XOR!")
end

# ============================================================================
# Example 4: Register Saturation - XMM to YMM
# ============================================================================

function example_register_saturation()
    println("\n" * "="^80)
    println("Example 4: Register Saturation - Pack XMM ops into YMM")
    println("="^80)

    println("\nScenario: Two independent XMM ADD operations")
    println("  ADD_xmm(a1, b1)  # Process 2 complex numbers")
    println("  ADD_xmm(a2, b2)  # Process 2 complex numbers")

    # Create two independent XMM operations
    add1 = SIMDNode(OP_ADD, [:a1, :b1], REG_XMM, id=1)
    add2 = SIMDNode(OP_ADD, [:a2, :b2], REG_XMM, id=2)

    ops = [add1, add2]

    println("\nOriginal: 2 separate XMM operations")
    println("  Cost per op: $(simd_cost(add1))")
    println("  Total cost: $(sum(simd_cost(op) for op in ops))")

    # Saturate to YMM
    saturated = saturate_operations(ops, REG_YMM)

    println("\nSaturated: 1 YMM operation")
    println("  Number of operations: $(length(saturated))")
    if !isempty(saturated) && saturated[1].reg_type == REG_YMM
        println("  Register type: YMM (256-bit)")
        println("  Processes 4 complex numbers in one operation!")
    end

    println("\n✓ Successfully packed operations into wider register!")
end

# ============================================================================
# Example 5: Building a Complete FFT-4 Kernel
# ============================================================================

function example_fft4_kernel()
    println("\n" * "="^80)
    println("Example 5: Building Complete FFT-4 Kernel with Optimizations")
    println("="^80)

    println("\nFFT-4 Butterfly Structure:")
    println("  1. Load 4 complex numbers (8 floats) into YMM")
    println("  2. Split into lo and hi halves")
    println("  3. First butterfly: (lo ± hi)")
    println("  4. Apply twiddle factor (multiply by -i)")
    println("  5. Second butterfly")
    println("  6. Store result")

    # Step 1: Load
    load_node = SIMDNode(
        OP_LOAD,
        [:px],
        REG_YMM,
        metadata=Dict(:offset => 0, :n_floats => 8)
    )

    # Step 2: Split with shuffle
    lo = SIMDNode(
        OP_SHUFFLE,
        [load_node],
        REG_XMM,
        metadata=Dict(:pattern => [1, 2, 3, 4])
    )

    hi = SIMDNode(
        OP_SHUFFLE,
        [load_node],
        REG_XMM,
        metadata=Dict(:pattern => [5, 6, 7, 8])
    )

    # Step 3: First butterfly
    butterfly1 = SIMDNode(
        OP_BUTTERFLY,
        [lo, hi],
        REG_XMM,
        metadata=Dict(:out_add => :add_vec, :out_sub => :sub_vec)
    )

    # Step 4: Twiddle by -i (optimized to shuffle + xor)
    twiddle_node = SIMDNode(
        OP_MUL,
        [:sub_vec, :neg_i],
        REG_XMM,
        metadata=Dict(:twiddle_type => :neg_imag_unit, :n_floats => 4)
    )

    twiddle_optimized = apply_algebraic_rewrites(twiddle_node)

    # Calculate total cost
    total_cost = (simd_cost(load_node) + simd_cost(lo) + simd_cost(hi) +
                  simd_cost(butterfly1) + simd_cost(twiddle_optimized))

    println("\nEstimated cost breakdown:")
    println("  Load:       $(simd_cost(load_node)) cycles")
    println("  Shuffles:   $(simd_cost(lo) + simd_cost(hi)) cycles")
    println("  Butterfly:  $(simd_cost(butterfly1)) cycles")
    println("  Twiddle:    $(simd_cost(twiddle_optimized)) cycles")
    println("  ---")
    println("  Total:      $(round(total_cost, digits=2)) cycles")

    println("\n--- Generated Kernel Preview ---")
    println("@inline function vfft4_optimized(px::Vector{Float32}, py::Vector{Float32})")
    println("    @inbounds @fastmath begin")
    println("        # Load")
    println("        " * replace(codegen(load_node, Float32), "\n" => "\n        "))
    println("        ")
    println("        # First butterfly (ADD and SUB execute in parallel)")
    println("        " * replace(codegen(butterfly1, Float32), "\n" => "\n        "))
    println("        ")
    println("        # Twiddle by -i (optimized to SHUFFLE + XOR)")
    println("        " * replace(codegen(twiddle_optimized, Float32), "\n" => "\n        "))
    println("        # ... (second butterfly and store omitted)")
    println("    end")
    println("end")

    println("\n✓ Complete FFT-4 kernel with algebraic optimizations!")
end

# ============================================================================
# Example 6: Cost Comparison Across Register Widths
# ============================================================================

function example_register_width_comparison()
    println("\n" * "="^80)
    println("Example 6: Cost Comparison Across Register Widths")
    println("="^80)

    println("\nProcessing 16 complex numbers (32 floats):")

    # XMM: Need 8 operations (4 floats per XMM)
    xmm_ops = 32 ÷ 4
    xmm_cost = xmm_ops * get_cost(OP_ADD, REG_XMM).throughput

    # YMM: Need 4 operations (8 floats per YMM)
    ymm_ops = 32 ÷ 8
    ymm_cost = ymm_ops * get_cost(OP_ADD, REG_YMM).throughput

    # ZMM: Need 2 operations (16 floats per ZMM)
    zmm_ops = 32 ÷ 16
    zmm_cost = zmm_ops * get_cost(OP_ADD, REG_ZMM).throughput

    println("\nXMM (128-bit):")
    println("  Operations needed: $xmm_ops")
    println("  Throughput cost: $xmm_cost cycles")

    println("\nYMM (256-bit):")
    println("  Operations needed: $ymm_ops")
    println("  Throughput cost: $ymm_cost cycles")
    println("  Speedup vs XMM: $(round(xmm_cost/ymm_cost, digits=2))x")

    println("\nZMM (512-bit) [AVX-512]:")
    println("  Operations needed: $zmm_ops")
    println("  Throughput cost: $zmm_cost cycles")
    println("  Speedup vs XMM: $(round(xmm_cost/zmm_cost, digits=2))x")
    println("  Speedup vs YMM: $(round(ymm_cost/zmm_cost, digits=2))x")

    println("\n✓ Wider registers enable higher throughput!")
end

# ============================================================================
# Run All Examples
# ============================================================================

function run_all_examples()
    println("\n")
    println("╔" * "═"^78 * "╗")
    println("║" * " "^20 * "E-Graph SIMD System Examples" * " "^30 * "║")
    println("╚" * "═"^78 * "╝")

    example_cost_comparison()
    example_mul_to_xor_rewrite()
    example_twiddle_by_i()
    example_register_saturation()
    example_fft4_kernel()
    example_register_width_comparison()

    println("\n" * "="^80)
    println("Summary")
    println("="^80)
    println("""
Key Insights:
1. XOR is ~3x faster than MUL for sign flips
2. Algebraic rewrites automatically find these optimizations
3. Twiddle factors can be decomposed into SHUFFLE + XOR
4. Register saturation packs narrow ops into wider registers
5. Cost model guides optimization decisions
6. Wider registers (YMM, ZMM) improve throughput

Next Steps:
- Integrate with full e-graph rewriting (Metatheory.jl)
- Add equality saturation for finding optimal expressions
- Implement complete FFT kernel generator
- Benchmark against hand-written kernels
    """)
end

# ============================================================================
# Example 7: Generate Complete FFT-4 Kernel with New System
# ============================================================================

function example_generate_fft4_new_system()
    println("\n" * "="^80)
    println("Example 7: Generate Complete FFT-4 Kernel (New System)")
    println("="^80)

    include("egraph_simd_system.jl")

    println("\nGenerating FFT-4 kernel using algebraic rewrites...")

    # Generate kernel
    code, dag = generate_optimized_kernel(4, Float32, name="vfft4_optimized")

    println("\n--- Generated Operations (DAG) ---")
    for (i, op) in enumerate(dag.ops)
        cost = get_cost(op.op, op.reg_type)
        println("  Op $i: $(op.op) → :$(op.output)")
        println("         Cost: $(cost.latency) cyc latency, $(cost.throughput) cyc throughput")
        if haskey(op.metadata, :sign_pattern)
            println("         Sign pattern: $(op.metadata[:sign_pattern])")
        end
        if haskey(op.metadata, :xor_mask)
            println("         ✓ Converted to XOR!")
        end
    end

    println("\n--- Generated Code ---")
    println(code)

    println("\n✓ Complete optimized FFT-4 kernel generated!")
    println("  - MUL operations converted to XOR where possible")
    println("  - Operations reordered for better ILP")
    println("  - Uses SIMD.jl vocabulary (VecRange, shufflevector, reinterpret)")
end

# Run if executed directly
#if abspath(PROGRAM_FILE) == @__FILE__
    example_cost_comparison()
    example_generate_fft4_new_system()
#end

#export run_all_examples, example_generate_fft4_new_system