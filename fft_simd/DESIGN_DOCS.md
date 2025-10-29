# SIMD FFT Kernel Generator: Design Documentation

## Overview

This system automates the generation of optimal SIMD FFT kernels by:
1. **Tracking** operations from the recursive n=2 base case
2. **Building** a computation DAG (Directed Acyclic Graph)
3. **Saturating** operations at appropriate levels into wide SIMD registers
4. **Generating** optimized Julia code with minimal cycles

## Core Concepts

### 1. SIMDOp Structure

```julia
struct SIMDOp
    op_type::OpType           # What operation (ADD, SUB, SHUFFLE, etc.)
    id::Int                   # Unique identifier
    inputs::Vector{Union{Int,Symbol}}  # Input operands
    output::Union{Int,Symbol} # Output operand
    register_size::RegisterSize  # XMM (128), YMM (256), ZMM (512)
    working_size::Int         # Number of COMPLEX elements (not floats!)
    layout::DataLayout        # How data is arranged in register
    metadata::Dict{Symbol,Any}  # Operation-specific data
    dependencies::Vector{Int} # Operations this depends on
    level::Int                # Recursion depth (0 = base case n=2)
end
```

**Key Fields:**
- `working_size`: Number of complex numbers, NOT floats. For 4 complex numbers = 8 floats
- `level`: Tracks recursion depth. Level 0 = n=2, Level 1 = n=4, Level 2 = n=8, etc.
- `metadata`: Flexible storage for operation-specific info

### 2. OpType Enumeration

```julia
@enum OpType begin
    OP_ADD          # a + b (vertical addition)
    OP_SUB          # a - b (vertical subtraction)
    OP_MUL          # a * b (pointwise multiplication)
    OP_COMPLEX_MUL  # (a+bi)*(c+di) - full complex multiply
    OP_SHUFFLE      # Permutation/rearrangement
    OP_TWIDDLE      # Twiddle factor application
    OP_SIGNFLIP     # Sign flip via XOR (for multiply by -1)
    OP_LOAD         # Memory load
    OP_STORE        # Memory store
    OP_NOP          # No-op (for padding)
end
```

### 3. Saturation Strategy

**The Problem:**
At n=2, we generate scalar operations (_r, _i suffixes). We need to pack these into wider registers.

**The Solution:**
```
Level 0 (n=2): [ADD_scalar, SUB_scalar] × 4 instances
                ↓ detect pattern
Level 1 (n=4): Pack into [ADD_xmm, SUB_xmm] (4 floats each)
                ↓ detect pattern  
Level 2 (n=8): Pack into [ADD_ymm, SUB_ymm] (8 floats each)
```

**Saturation Rules:**
1. **Same operation type**: Can only fuse ADD with ADD, SUB with SUB
2. **Same level**: Operations must be from same recursion level
3. **Fits in register**: Total floats ≤ register width
4. **Butterfly pattern**: Paired ADD/SUB with same inputs → single fused op

## Integration with Existing Code

### Current recfft2_simd (String-based)

Your current code generates strings like:
```julia
"x1_r, x1_i = real(x[1]), imag(x[1])"
"t1_r, t1_i = x1_r + x3_r, x1_i + x3_i"
```

### New Approach (DAG-based)

Replace string generation with operation tracking:

```julia
function recfft2_simd_dag(n, x, T; level=0, tmp_base=1)
    dag = OperationDAG()
    
    if n == 2
        # Instead of: "t1 = x1 + x2"
        # Create:
        add_op!(dag, SIMDOp(
            OP_ADD, -1, 
            [Symbol(x[1]), Symbol(x[2])],  # inputs
            Symbol("t", tmp_base),          # output
            register_size=REG_XMM,
            working_size=1,                 # 1 complex number
            level=level
        ))
        
        # Instead of: "t2 = x1 - x2"  
        add_op!(dag, SIMDOp(
            OP_SUB, -1,
            [Symbol(x[1]), Symbol(x[2])],
            Symbol("t", tmp_base+1),
            register_size=REG_XMM,
            working_size=1,
            level=level
        ))
        
        return dag
    else
        # Recursive case: merge sub-DAGs
        dag_even = recfft2_simd_dag(n÷2, x[1:2:n], T; level=level+1, ...)
        dag_odd = recfft2_simd_dag(n÷2, x[2:2:n], T; level=level+1, ...)
        
        # Merge
        for op in dag_even.ops
            add_op!(dag, op)
        end
        for op in dag_odd.ops
            add_op!(dag, op)
        end
        
        # Add butterfly combining operations
        # ...
        
        return dag
    end
end
```

### Saturation Point

At the appropriate recursion level (when `working_size` × num_ops fits in YMM):

```julia
# After building DAG, before code generation
function apply_saturation!(dag::OperationDAG)
    # Find operations at each level
    for level in reverse(0:max_level)
        level_ops = filter(op -> op.level == level, dag.ops)
        
        # Detect butterfly patterns
        butterflies = find_butterfly_pairs(level_ops)
        
        if can_fit_in_ymm(butterflies)
            # Replace multiple scalar ops with single YMM op
            fused = saturate_butterfly(butterflies, REG_YMM)
            replace_ops!(dag, butterflies, fused)
        end
    end
end
```

## Example: vfft4 Generation

### Step 1: Build DAG (n=4)

```
Recursion tree:
    n=4 (level 0)
    ├── n=2 even (level 1): x1, x3
    │   ├── ADD: x1 + x3 → t1
    │   └── SUB: x1 - x3 → t2
    └── n=2 odd (level 1): x2, x4
        ├── ADD: x2 + x4 → t3
        └── SUB: x2 - x4 → t4
    
    Butterfly (level 0):
    ├── ADD: t1 + t3 → y1
    ├── SUB: t1 - t3 → y3
    ├── ADD: t2 + t4*twiddle → y2
    └── SUB: t2 - t4*twiddle → y4
```

### Step 2: Detect Patterns

At level 1:
- 2 × ADD operations (working_size=1 each)
- 2 × SUB operations (working_size=1 each)
- Total: 4 complex = 8 floats → fits in YMM!

Pattern: Butterfly pair (ADD + SUB with same inputs)

### Step 3: Saturate

Replace:
```
ADD: x1 + x3 → t1 (XMM, 2 floats)
SUB: x1 - x3 → t2 (XMM, 2 floats)
ADD: x2 + x4 → t3 (XMM, 2 floats)
SUB: x2 - x4 → t4 (XMM, 2 floats)
```

With:
```
LOAD: [x1, x2, x3, x4] → v (YMM, 8 floats)
SHUFFLE_SPLIT: v → lo=[x1,x2], hi=[x3,x4] (2×XMM, 4 floats each)
BUTTERFLY: lo ± hi → add_vec, sub_vec (2×XMM, 4 floats each)
```

### Step 4: Generate Code

```julia
@inline function vfft4_generated(px::Vector{Float32}, py::Vector{Float32})
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = px[LANE + 1]
        
        lo = shufflevector(v, Val((0, 1, 2, 3)))
        hi = shufflevector(v, Val((4, 5, 6, 7)))
        
        add_vec = lo + hi
        sub_vec = lo - hi
        
        # Twiddle application
        sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2))) * 
                  Vec{4,Float32}((1, 1, 1, -1))
        
        # Second butterfly
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))
        
        y12 = t12 + t34
        y34 = t12 - t34
        
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        py[LANE + 1] = OUT
    end
end
```

## Advanced: Twiddle Factor Handling

### Problem

At n=4, second element needs twiddle: `(x2-x4) * i`

### Solution in DAG

```julia
# Detect twiddle pattern in metadata
if op.metadata[:twiddle] == im
    # Option 1: Shuffle + sign flip (XOR)
    shuffle_op = SIMDOp(OP_SHUFFLE, ..., 
        metadata=Dict(:shuffle_pattern => [1,2,4,3]))  # swap r,i
    signflip_op = SIMDOp(OP_SIGNFLIP, ...,
        metadata=Dict(:sign_pattern => (0,0,0,0x80000000)))  # negate i
    
    # Option 2: Explicit multiply
    mul_op = SIMDOp(OP_TWIDDLE, ...,
        metadata=Dict(:twiddle_expr => "Vec{4}((1,1,1,-1))"))
end
```

### Code Generation

```julia
function generate_julia_code(op::SIMDOp, T)
    if op.op_type == OP_TWIDDLE
        twiddle = op.metadata[:twiddle]
        if twiddle == im
            # Generate shuffle + signflip
            return """
            $(op.output)_shuffled = shufflevector($(op.inputs[1]), Val((0,1,3,2)))
            $(op.output) = signflip($(op.output)_shuffled, 0x80000000)
            """
        else
            # General twiddle
            return "$(op.output) = $(op.inputs[1]) * $(twiddle_to_vec(twiddle))"
        end
    end
    # ... other cases
end
```

## Optimization Opportunities

### 1. Shuffle Fusion

Detect chains:
```
shuffle1 → shuffle2 → shuffle3
```

Fuse into single shuffle:
```
shuffle_combined (pattern = compose_patterns(p1, p2, p3))
```

### 2. Memory Access Coalescing

Detect:
```
LOAD(offset=0, size=2)
LOAD(offset=4, size=2)
```

Replace with:
```
LOAD(offset=0, size=4)  # Single wide load
```

### 3. Butterfly Parallelization

Recognize that `a+b` and `a-b` can execute in parallel:
```
# Instead of sequential:
tmp1 = a + b
tmp2 = a - b

# Generate parallel-friendly:
tmp_both = butterfly(a, b)  # Returns both results
```

## Usage Example

```julia
# 1. Build DAG from your recursive generator
dag = recfft2_to_dag(8, Float32)

# 2. Print for inspection
print_dag(dag)

# 3. Apply saturation
apply_saturation!(dag, target_register=REG_YMM)

# 4. Generate optimized code
code = generate_kernel_code(dag, Float32, "vfft8_optimized")

# 5. Evaluate and use
eval(Meta.parse(code))
x = rand(Float32, 16)  # 8 complex as 16 floats
y = similar(x)
vfft8_optimized(x, y)
```

## Next Steps

### TODO: Complete Implementation

1. **Finish `apply_saturation!` function**
   - Implement saturation rules
   - Handle all operation types
   - Optimize shuffle patterns

2. **Enhance pattern detection**
   - Add more fusion patterns
   - Detect memory access patterns
   - Recognize special cases (power-of-2, etc.)

3. **Improve code generation**
   - Generate cleaner code
   - Add comments explaining operations
   - Optimize register allocation

4. **Add benchmarking**
   - Compare generated kernels against hand-written
   - Measure cycle counts
   - Validate correctness

5. **Extend to larger radices**
   - Support n=8, 16, 32, ...
   - Handle split-radix algorithms
   - Mixed-radix support

## Benefits of This Approach

1. **Correctness**: DAG ensures dependencies are tracked
2. **Optimization**: Saturation happens at the right level automatically  
3. **Flexibility**: Easy to add new operations and patterns
4. **Debugging**: Can visualize and inspect the computation graph
5. **Portability**: Same DAG can generate code for different SIMD ISAs
6. **Maintainability**: Separate concerns (algorithm vs. codegen)

## Comparison: String-based vs DAG-based

| Aspect | String-based | DAG-based |
|--------|--------------|-----------|
| Correctness | Error-prone (typos, wrong vars) | Type-safe, validated |
| Optimization | Hard to detect patterns in strings | Easy pattern matching on ops |
| Debugging | Print strings, manual inspection | Visualize graph, inspect nodes |
| Portability | Code generation tied to Julia | Can target multiple backends |
| Composability | Concatenate strings | Merge graphs |
| Analysis | Parse strings (fragile) | Graph algorithms |

