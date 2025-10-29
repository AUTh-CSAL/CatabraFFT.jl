# Quick Start: SIMDOp-Based FFT Kernel Generator

## TL;DR

The `SIMDOp` system lets you automatically generate optimal SIMD FFT kernels by:
1. **Tracking** operations from recursive n=2 base cases
2. **Saturating** them into wide registers at the right level
3. **Generating** optimized code with minimal shuffles/cycles

## Key Concepts in 60 Seconds

### SIMDOp Struct
```julia
struct SIMDOp
    op_type::OpType           # ADD, SUB, SHUFFLE, LOAD, STORE, etc.
    working_size::Int         # Number of COMPLEX elements (not floats!)
    level::Int                # Recursion depth (0=n=2, 1=n=4, ...)
    register_size::RegisterSize  # XMM(128), YMM(256), ZMM(512)
    metadata::Dict            # Extra info (shuffle patterns, twiddles, etc.)
    # ... other fields for tracking dependencies
end
```

**Important**: `working_size` counts **complex numbers**, not floats!
- `working_size=2` → 4 floats → XMM register
- `working_size=4` → 8 floats → YMM register

### The Big Idea

Instead of generating code strings like:
```julia
"x1_r, x1_i = real(x[1]), imag(x[1])"
"t1_r, t1_i = x1_r + x3_r, x1_i + x3_i"
```

Build a graph of operations:
```julia
add_op!(dag, SIMDOp(OP_ADD, ..., inputs=[:x1, :x3], output=:t1, working_size=1))
```

Then at the right level, **saturate** multiple scalar ops into one wide SIMD op!

## Files Overview

1. **`simd_op_system.jl`** - Core system
   - `SIMDOp` struct definition
   - `OperationDAG` for tracking all operations
   - Saturation functions (`can_saturate`, `saturate_butterfly`)
   - Code generation (`generate_julia_code`, `generate_kernel_code`)

2. **`simd_op_examples.jl`** - Usage examples
   - Manual DAG construction for vfft4
   - Saturation demonstration
   - Integration with recursive generator

3. **`DESIGN_DOCS.md`** - Detailed design documentation
   - Explanation of all concepts
   - vfft4 generation walkthrough
   - Optimization opportunities

4. **`INTEGRATION_GUIDE.md`** - How to integrate with your code
   - Phase 1: Hybrid (track + generate strings)
   - Phase 2: Pure DAG (no strings)
   - Phase 3: Full automation

## Quick Example: Generate vfft4

```julia
include("simd_op_system.jl")

# Create DAG manually
dag = OperationDAG()

# Load operation
add_op!(dag, SIMDOp(OP_LOAD, -1, [], :v,
    register_size=REG_YMM, working_size=4))

# Split
lo_id = add_op!(dag, SIMDOp(OP_SHUFFLE, -1, [:v], :lo,
    register_size=REG_XMM, working_size=2,
    metadata=Dict(:shuffle_pattern => [1,2,3,4])))

hi_id = add_op!(dag, SIMDOp(OP_SHUFFLE, -1, [:v], :hi,
    register_size=REG_XMM, working_size=2,
    metadata=Dict(:shuffle_pattern => [5,6,7,8])))

# Butterfly
add_op!(dag, SIMDOp(OP_ADD, -1, [:lo, :hi], :add_vec,
    dependencies=[lo_id, hi_id]))

add_op!(dag, SIMDOp(OP_SUB, -1, [:lo, :hi], :sub_vec,
    dependencies=[lo_id, hi_id]))

# ... (continue with twiddle, second butterfly, store)

# Generate code
code = generate_kernel_code(dag, Float32, "vfft4")
println(code)
```

## Integration Roadmap

### Step 1: Understand (1 hour)
- Read `DESIGN_DOCS.md`
- Run examples from `simd_op_examples.jl`
- Inspect generated DAGs with `print_dag(dag)`

### Step 2: Hybrid Approach (2-3 hours)
- Modify your `recfft2_simd` to track operations
- Keep generating strings (for validation)
- Build DAG alongside
- Compare DAG structure at different recursion levels

### Step 3: Analyze (1 hour)
- Use `detect_patterns(dag, level)` to find saturation opportunities
- Check which levels can fit in YMM registers
- Identify butterfly patterns

### Step 4: Pure DAG (4-5 hours)
- Replace string generation with DAG building
- Implement saturation rules
- Generate code from DAG
- Validate against FFTW

### Step 5: Optimize (ongoing)
- Add more saturation patterns
- Implement shuffle fusion
- Optimize memory access
- Extend to larger radices

## Common Patterns

### Pattern 1: Butterfly (Parallel ADD/SUB)
```julia
# Instead of:
ADD: a + b → t1
SUB: a - b → t2

# Recognize as:
BUTTERFLY: a, b → (t1=a+b, t2=a-b)
# CPU can execute both in parallel!
```

### Pattern 2: Twiddle by i
```julia
# Instead of:
MUL: (a+bi) * i

# Use:
SHUFFLE: [a, b] → [b, a]
SIGNFLIP: [b, a] → [b, -a]  (XOR with 0x80000000)
# Much faster than multiplication!
```

### Pattern 3: Saturation
```julia
# 4 scalar operations at level 1:
OP1: x1 + x3 → t1 (working_size=1, level=1)
OP2: x1 - x3 → t2 (working_size=1, level=1)
OP3: x2 + x4 → t3 (working_size=1, level=1)
OP4: x2 - x4 → t4 (working_size=1, level=1)

# Saturate into 2 XMM operations:
OP_SAT1: [x1,x2] + [x3,x4] → [t1,t3] (working_size=2, REG_XMM)
OP_SAT2: [x1,x2] - [x3,x4] → [t2,t4] (working_size=2, REG_XMM)
```

## Debugging Tips

### Visualize DAG
```julia
print_dag(dag)
# Shows all operations with dependencies
```

### Check Saturation Opportunities
```julia
for level in 0:max_level
    level_ops = filter(op -> op.level == level, dag.ops)
    println("Level $level: $(length(level_ops)) ops")
    
    if can_saturate(level_ops, REG_YMM)
        println("  ✓ Can saturate to YMM")
        total_floats = sum(op.working_size * 2 for op in level_ops)
        println("  Total: $total_floats floats")
    end
end
```

### Validate Generated Code
```julia
# Compare against reference
x = rand(Float32, 8)
y_ref = similar(x)
y_gen = similar(x)

vfft4_manual(x, y_ref)    # Your hand-written kernel
vfft4_generated(x, y_gen)  # Auto-generated kernel

@assert isapprox(y_ref, y_gen, rtol=1e-5)
```

## Next Steps

1. **Read** `DESIGN_DOCS.md` for complete explanation
2. **Try** examples in `simd_op_examples.jl`
3. **Integrate** using `INTEGRATION_GUIDE.md` (start with Phase 1)
4. **Experiment** with different saturation strategies
5. **Extend** to larger FFT sizes

## Key Advantages

✅ **Correctness**: Type-safe operation tracking
✅ **Optimization**: Automatic saturation at optimal levels  
✅ **Debugging**: Visualize computation graph
✅ **Portability**: Same DAG → multiple targets (AVX2, AVX-512, NEON)
✅ **Maintainability**: Separate algorithm from codegen

## Questions?

The system is designed to be:
- **Incremental**: Start with Phase 1 (hybrid), migrate gradually
- **Inspectable**: `print_dag()` shows what's happening
- **Extensible**: Easy to add new operations and patterns
- **Validatable**: Compare against FFTW or hand-written kernels

Read the docs, run the examples, and start integrating! 🚀

