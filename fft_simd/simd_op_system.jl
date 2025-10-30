# ============================================================================
# SIMD Operation Tracking System for Automated FFT Kernel Generation
# ============================================================================

using SIMD

# ============================================================================
# Core Data Structures
# ============================================================================

"""
    OpType

Enumeration of fundamental SIMD operations.
These are the atomic operations discovered at the n=2 recursion level.
"""
@enum OpType begin
    OP_ADD          # Vertical addition: a + b
    OP_SUB          # Vertical subtraction: a - b
    OP_MUL          # Pointwise multiplication: a * b
    OP_COMPLEX_MUL  # Complex multiplication: (a+bi)*(c+di)
    OP_SHUFFLE      # Permutation/shuffle
    OP_TWIDDLE      # Twiddle factor application
    OP_SIGNFLIP     # Sign flip via XOR
    OP_LOAD         # Load from memory
    OP_STORE        # Store to memory
    OP_NOP          # No operation (for padding)
end

"""
    DataLayout

How the complex data is laid out in the SIMD register.
"""
@enum DataLayout begin
    LAYOUT_INTERLEAVED    # [r1, i1, r2, i2, r3, i3, r4, i4]
    LAYOUT_SPLIT_RI       # Real and imaginary in separate vectors
    LAYOUT_COMPLEX_PAIRS  # Grouped complex pairs: [r1,i1,r2,i2] [r3,i3,r4,i4]
end

"""
    RegisterSize

Target register size for SIMD operations.
"""
@enum RegisterSize begin
    REG_XMM = 128    # 128-bit SSE4 registers (4 x Float32)
    REG_YMM = 256    # 256-bit AVX2 registers (8 x Float32)
    REG_ZMM = 512    # 512-bit AVX-512/AVX-10 registers (16 x Float32)
end

"""
    SIMDOp

Represents a single SIMD operation in the computation DAG.

# Fields
- `op_type::OpType`: The operation type
- `id::Int`: Unique identifier for this operation
- `inputs::Vector{Union{Int,Symbol}}`: Input operand IDs or symbols (e.g., :x1, :x2)
- `output::Union{Int,Symbol}`: Output operand ID or symbol
- `register_size::RegisterSize`: Target register size
- `working_size::Int`: Number of complex elements (not floats)
- `layout::DataLayout`: Data layout in register
- `metadata::Dict{Symbol,Any}`: Additional operation-specific data
    - :shuffle_pattern => Vector{Int}
    - :twiddle_factors => Vector{ComplexF16/32/64}
    - :sign_mask => UInt32 bitmask
    - :memory_offset => Int
    - :is_contiguous => Bool
- `dependencies::Vector{Int}`: Operation IDs this depends on
- `level::Int`: Recursion level where this operation was created (0 = base case n=2)
"""
mutable struct SIMDOp
    op_type::OpType
    id::Int
    inputs::Vector{Union{Int,Symbol}}
    output::Union{Int,Symbol}
    register_size::RegisterSize
    working_size::Int  # Number of complex elements
    layout::DataLayout
    metadata::Dict{Symbol,Any}
    dependencies::Vector{Int}
    level::Int
    
    # Constructor with defaults
    function SIMDOp(op_type::OpType, id::Int, inputs::Vector, output;
                    register_size=REG_XMM, working_size=2, layout=LAYOUT_INTERLEAVED,
                    metadata=Dict{Symbol,Any}(), dependencies=Int[], level=0)
        new(op_type, id, inputs, output, register_size, working_size, layout,
            metadata, dependencies, level)
    end
end

# ============================================================================
# Operation DAG Management
# ============================================================================

"""
    OperationDAG

Directed Acyclic Graph representing the complete computation.
"""
mutable struct OperationDAG
    ops::Vector{SIMDOp}
    op_counter::Int
    symbol_map::Dict{Symbol,Int}  # Maps symbols to operation IDs
    
    OperationDAG() = new(SIMDOp[], 0, Dict{Symbol,Int}())
end

"""
    add_op!(dag, op) -> Int

Add an operation to the DAG and return its ID.
"""
function add_op!(dag::OperationDAG, op::SIMDOp)
    dag.op_counter += 1
    op.id = dag.op_counter
    push!(dag.ops, op)
    
    # Register output in symbol map
    if op.output isa Symbol
        dag.symbol_map[op.output] = op.id
    end
    
    return op.id
end

"""
    get_op_by_symbol(dag, sym) -> Union{SIMDOp, Nothing}

Retrieve an operation by its output symbol.
"""
function get_op_by_symbol(dag::OperationDAG, sym::Symbol)
    id = get(dag.symbol_map, sym, nothing)
    return id === nothing ? nothing : dag.ops[id]
end

# ============================================================================
# Operation Saturation/Fusion
# ============================================================================

"""
    SaturationRule

Rule for combining multiple operations into a single wide SIMD operation.
"""
struct SaturationRule
    name::Symbol
    pattern::Function  # (ops::Vector{SIMDOp}) -> Bool
    fuse::Function     # (ops::Vector{SIMDOp}) -> SIMDOp
    priority::Int
end

"""
    can_saturate(ops::Vector{SIMDOp}, target_size::RegisterSize) -> Bool

Check if a set of operations can be saturated into a wider register.
"""
function can_saturate(ops::Vector{SIMDOp}, target_size::RegisterSize)
    if isempty(ops)
        return false
    end
    
    # All must be same operation type
    if length(unique(op.op_type for op in ops)) > 1
        println("Not all ops have the same operation type")
        return false
    end
    
    # All must have same level
    if length(unique(op.level for op in ops)) > 1
        println("Not all ops have the same level")
        return false
    end
    
    # Check total size fits in target register
    total_floats = sum(op.working_size * 2 for op in ops)
    max_floats = Int(target_size) ÷ 32  # Assuming Float32
    @show total_floats, max_floats
    
    return total_floats <= max_floats
end

"""
    saturate_add_sub_ops(ops::Vector{SIMDOp}, target_size::RegisterSize) -> SIMDOp

Saturate multiple ADD/SUB operations into a single wide operation.
"""
function saturate_add_sub_ops(ops::Vector{SIMDOp}, target_size::RegisterSize)
    @assert all(op.op_type in [OP_ADD, OP_SUB] for op in ops)
    
    # Collect all inputs and build shuffle pattern
    all_inputs = []
    shuffle_pattern = Int[]
    current_offset = 0
    
    for (i, op) in enumerate(ops)
        push!(all_inputs, op.inputs...)
        # Add shuffle indices for this operation's contribution
        for j in 1:(op.working_size * 2)
            push!(shuffle_pattern, current_offset + j - 1)
        end
        current_offset += op.working_size * 2
    end
    
    total_working_size = sum(op.working_size for op in ops)
    
    # Create fused operation
    fused_op = SIMDOp(
        ops[1].op_type,
        -1,  # Will be assigned by DAG
        unique(all_inputs),
        :fused_result,
        target_size,
        total_working_size,
        LAYOUT_INTERLEAVED,
        Dict(:shuffle_pattern => shuffle_pattern,
             :source_ops => [op.id for op in ops]),
        Int[],
        ops[1].level
    )
    
    return fused_op
end

"""
    saturate_butterfly(ops::Vector{SIMDOp}, target_size::RegisterSize) -> Vector{SIMDOp}

Saturate a butterfly pattern (parallel ADD and SUB) into minimal operations.
"""
function saturate_butterfly(ops::Vector{SIMDOp}, target_size::RegisterSize)
    # Separate ADD and SUB operations
    add_ops = filter(op -> op.op_type == OP_ADD, ops)
    sub_ops = filter(op -> op.op_type == OP_SUB, ops)
    
    if length(add_ops) != length(sub_ops)
        return ops  # Can't optimize
    end
    
    # Group by common inputs
    butterfly_groups = []
    for (add_op, sub_op) in zip(add_ops, sub_ops)
        if add_op.inputs == sub_op.inputs
            push!(butterfly_groups, (add_op, sub_op))
        end
    end
    
    if length(butterfly_groups) != length(add_ops)
        return ops  # Can't optimize all
    end
    
    # Create single butterfly operation that does both ADD and SUB
    # This is more efficient as the CPU can execute them in parallel
    total_working_size = sum(add_op.working_size for add_op in add_ops)
    
    butterfly_op = SIMDOp(
        OP_ADD,  # We'll handle both in code generation
        -1,
        unique(vcat([op.inputs for op in add_ops]...)),
        :butterfly_result,
        target_size,
        total_working_size,
        LAYOUT_INTERLEAVED,
        Dict(:is_butterfly => true,
             :add_ops => [op.id for op in add_ops],
             :sub_ops => [op.id for op in sub_ops]),
        Int[],
        add_ops[1].level
    )
    
    return [butterfly_op]
end

# ============================================================================
# Operation Pattern Recognition
# ============================================================================

"""
    detect_patterns(dag::OperationDAG, level::Int) -> Vector{Vector{Int}}

Detect fusible operation patterns at a given recursion level.
Returns groups of operation IDs that can be fused.
"""
function detect_patterns(dag::OperationDAG, level::Int)
    level_ops = filter(op -> op.level == level, dag.ops)
    
    patterns = Vector{Int}[]
    
    # Pattern 1: Consecutive ADD operations
    add_ops = filter(op -> op.op_type == OP_ADD, level_ops)
    if length(add_ops) >= 2
        push!(patterns, [op.id for op in add_ops])
    end
    
    # Pattern 2: Butterfly (paired ADD/SUB with same inputs)
    butterfly_pairs = []
    for op in level_ops
        if op.op_type == OP_ADD
            # Look for matching SUB with same inputs
            matching_sub = findfirst(o -> o.op_type == OP_SUB && o.inputs == op.inputs, level_ops)
            if matching_sub !== nothing
                push!(butterfly_pairs, (op.id, level_ops[matching_sub].id))
            end
        end
    end
    
    if !isempty(butterfly_pairs)
        push!(patterns, vcat([p[1] for p in butterfly_pairs], [p[2] for p in butterfly_pairs]))
    end
    
    # Pattern 3: Shuffle followed by operation
    for i in 1:(length(level_ops)-1)
        if level_ops[i].op_type == OP_SHUFFLE
            next_op = level_ops[i+1]
            if next_op.op_type in [OP_ADD, OP_SUB, OP_MUL]
                if level_ops[i].output in next_op.inputs
                    push!(patterns, [level_ops[i].id, next_op.id])
                end
            end
        end
    end
    
    return patterns
end

# ============================================================================
# Code Generation from SIMDOp
# ============================================================================

"""
    generate_julia_code(op::SIMDOp, T::Type) -> String

Generate Julia code for a single SIMD operation.
"""
function generate_julia_code(op::SIMDOp, T::Type)
    n_floats = op.working_size * 2
    
    if op.op_type == OP_LOAD
        if get(op.metadata, :is_contiguous, true)
            offset = get(op.metadata, :memory_offset, 0)
            return "LANE = VecRange{$n_floats}(0)\n$(op.output) = px[LANE + $(offset+1)]"
        else
            indices = op.metadata[:indices]
            return "$(op.output) = vgather(px, Vec($(tuple(indices...))))"
        end
        
    elseif op.op_type == OP_STORE
        return "py[LANE + 1] = $(op.inputs[1])"
        
    elseif op.op_type == OP_ADD
        if haskey(op.metadata, :is_butterfly) && op.metadata[:is_butterfly]
            # Generate both ADD and SUB in parallel
            in1, in2 = op.inputs[1], op.inputs[2]
            return """
            $(op.output)_add = $in1 + $in2
            $(op.output)_sub = $in1 - $in2
            """
        else
            in1, in2 = op.inputs[1], op.inputs[2]
            return "$(op.output) = $in1 + $in2"
        end
        
    elseif op.op_type == OP_SUB
        in1, in2 = op.inputs[1], op.inputs[2]
        return "$(op.output) = $in1 - $in2"
        
    elseif op.op_type == OP_SHUFFLE
        pattern = op.metadata[:shuffle_pattern]
        # Convert to 0-indexed for LLVM convention
        pattern_0idx = tuple((p-1 for p in pattern)...)
        return "$(op.output) = shufflevector($(op.inputs[1]), Val($pattern_0idx))"
        
    elseif op.op_type == OP_SIGNFLIP
        sign_pattern = get(op.metadata, :sign_mask, 0x80000000)
        mask_vec = "Vec{$n_floats,UInt32}($(op.metadata[:sign_pattern]))"
        # Possible slow-down due to immediate depedency between mask reinterpretation to uint and xors
        return """
        $(op.output)_uint = reinterpret(Vec{$n_floats,UInt32}, $(op.inputs[1]))
        $(op.output)_flipped = $(op.output)_uint ⊻ $mask_vec
        $(op.output) = reinterpret(Vec{$n_floats,$T}, $(op.output)_flipped)
        """
        
    elseif op.op_type == OP_TWIDDLE
        twiddle_expr = op.metadata[:twiddle_expr]
        return "$(op.output) = $(op.inputs[1]) * $twiddle_expr"
        
    elseif op.op_type == OP_MUL
        in1, in2 = op.inputs[1], op.inputs[2]
        return "$(op.output) = $in1 * $in2"
        
    else
        error("Unsupported operation type: $(op.op_type)")
    end
end

"""
    generate_kernel_code(dag::OperationDAG, T::Type, kernel_name::String) -> String

Generate complete Julia FFT kernel from operation DAG.
"""
function generate_kernel_code(dag::OperationDAG, T::Type, kernel_name::String)
    # Sort operations by dependencies (topological sort)
    sorted_ops = topological_sort(dag)
    
    # Generate code for each operation
    code_lines = String[]
    push!(code_lines, "@inline function $(kernel_name)(px::Vector{$T}, py::Vector{$T})")
    push!(code_lines, "    @inbounds @fastmath begin")
    
    for op in sorted_ops
        code = generate_julia_code(op, T)
        for line in split(code, '\n')
            push!(code_lines, "        " * line)
        end
    end
    
    push!(code_lines, "    end")
    push!(code_lines, "end")
    
    return join(code_lines, '\n')
end

"""
    topological_sort(dag::OperationDAG) -> Vector{SIMDOp}

Sort operations by dependencies.
"""
function topological_sort(dag::OperationDAG)
    sorted = SIMDOp[]
    visited = Set{Int}()
    
    function visit(op_id::Int)
        if op_id in visited
            return
        end
        
        op = dag.ops[op_id]
        for dep_id in op.dependencies
            visit(dep_id)
        end
        
        push!(sorted, op)
        push!(visited, op_id)
    end
    
    for op in dag.ops
        visit(op.id)
    end
    
    return sorted
end

# ============================================================================
# Integration with Recursive FFT Generator
# ============================================================================

"""
    recfft2_to_dag(n::Int, T::Type) -> OperationDAG

Convert recursive FFT decomposition to operation DAG.
This replaces the string-based code generation.
"""
function recfft2_to_dag(n::Int, T::Type; level=0, tmp_counter=Ref(0))
    dag = OperationDAG()
    
    if n == 1
        # Base case: identity
        return dag
        
    elseif n == 2
        # 2-point butterfly: the fundamental building block
        # x1, x2 -> y1 = x1 + x2, y2 = x1 - x2
        
        # Load inputs
        load1_id = add_op!(dag, SIMDOp(OP_LOAD, -1, [], :x1,
                                        working_size=1, level=level,
                                        metadata=Dict(:memory_offset => 0)))
        
        load2_id = add_op!(dag, SIMDOp(OP_LOAD, -1, [], :x2,
                                        working_size=1, level=level,
                                        metadata=Dict(:memory_offset => 2)))
        
        # Combine into single Vec{4} for 2 complex numbers
        combine_id = add_op!(dag, SIMDOp(OP_SHUFFLE, -1, [:x1, :x2], :x12,
                                          register_size=REG_XMM, working_size=2, level=level,
                                          metadata=Dict(:shuffle_pattern => [1,2,3,4])))
        
        # ADD operation: y1 = x1 + x2
        add_id = add_op!(dag, SIMDOp(OP_ADD, -1, [:x1, :x2], :y1,
                                      register_size=REG_XMM, working_size=1, level=level,
                                      dependencies=[load1_id, load2_id]))
        
        # SUB operation: y2 = x1 - x2
        sub_id = add_op!(dag, SIMDOp(OP_SUB, -1, [:x1, :x2], :y2,
                                      register_size=REG_XMM, working_size=1, level=level,
                                      dependencies=[load1_id, load2_id]))
        
        return dag
        
    else
        # Recursive case
        n2 = n ÷ 2
        
        # Recursively build DAG for even and odd elements
        dag_even = recfft2_to_dag(n2, T; level=level+1, tmp_counter=tmp_counter)
        dag_odd = recfft2_to_dag(n2, T; level=level+1, tmp_counter=tmp_counter)
        
        # Merge DAGs
        for op in dag_even.ops
            add_op!(dag, op)
        end
        for op in dag_odd.ops
            add_op!(dag, op)
        end
        
        # Add butterfly combination operations
        for i in 1:n2
            # Get temporary results from sub-FFTs
            t_even = Symbol("t", 2*i-1)
            t_odd = Symbol("t", 2*i)
            
            # Output indices
            y_plus = Symbol("y", i)
            y_minus = Symbol("y", i+n2)
            
            # Twiddle factor
            twiddle = cispi(-2*(i-1)/n)
            
            # y[i] = t_even + twiddle * t_odd
            add_op!(dag, SIMDOp(OP_ADD, -1, [t_even, t_odd], y_plus,
                                register_size=REG_XMM, working_size=1, level=level,
                                metadata=Dict(:twiddle => twiddle)))
            
            # y[i+n2] = t_even - twiddle * t_odd
            add_op!(dag, SIMDOp(OP_SUB, -1, [t_even, t_odd], y_minus,
                                register_size=REG_XMM, working_size=1, level=level,
                                metadata=Dict(:twiddle => twiddle)))
        end
        
        return dag
    end
end

# ============================================================================
# Example: Generate vfft4 automatically
# ============================================================================

function generate_vfft4(T::Type=Float32)
    dag = recfft2_to_dag(4, T)
    
    # Apply saturation at n=4 level
    patterns = detect_patterns(dag, 1)  # Level 1 is where n=4 operations are
    
    # TODO: Apply saturation rules
    
    # Generate code
    kernel_code = generate_kernel_code(dag, T, "vfft4_generated")
    
    return kernel_code
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    print_dag(dag::OperationDAG)

Pretty print the operation DAG for debugging.
"""
function print_dag(dag::OperationDAG)
    println("Operation DAG:")
    println("=" ^ 80)
    for op in dag.ops
        println("Op $(op.id): $(op.op_type)")
        println("  Inputs: $(op.inputs)")
        println("  Output: $(op.output)")
        println("  Size: $(op.working_size) complex, $(op.register_size)")
        println("  Level: $(op.level)")
        println("  Dependencies: $(op.dependencies)")
        if !isempty(op.metadata)
            println("  Metadata: $(op.metadata)")
        end
        println()
    end
end

export SIMDOp, OperationDAG, OpType, DataLayout, RegisterSize
export add_op!, detect_patterns, saturate_add_sub_ops, saturate_butterfly
export generate_julia_code, generate_kernel_code, recfft2_to_dag
export print_dag

