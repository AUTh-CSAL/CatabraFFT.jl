# ============================================================================
# SIMD FFT Kernel Generation with Algebraic Optimization
# ============================================================================
# Generates optimal SIMD code from recursive FFT decomposition.
# Uses algebraic rewrite rules to convert expensive operations to efficient ones.
# No Metatheory.jl dependency - simple pattern matching is sufficient and clearer.
# ============================================================================

using SIMD

# ============================================================================
# Core SIMD Operation Types
# ============================================================================

"""
SIMD operation types. Simplified - removed OP_COMPLEX_MUL and OP_TWIDDLE.
"""
@enum SIMDOp begin
    OP_ADD          # Vertical addition: a + b
    OP_SUB          # Vertical subtraction: a - b
    OP_MUL          # Pointwise multiplication (includes complex mul, twiddles)
    OP_XOR          # XOR for sign flips (much cheaper than MUL!)
    OP_SHUFFLE      # Permutation via shufflevector
    OP_LOAD         # Load from memory
    OP_STORE        # Store to memory
    OP_NOP          # No operation
end

"""
Register width types
"""
@enum RegType begin
    REG_XMM = 128      # SSE 128-bit (4 Float32 or 2 Float64)
    REG_YMM = 256      # AVX2 256-bit
    REG_ZMM = 512      # AVX-512 512-bit
end

# ============================================================================
# CPU Cycle Cost Model
# ============================================================================

"""
CPU cycle costs: (latency, reciprocal_throughput)
Based on Intel Optimization Reference Manual and uops.info measurements.
"""
struct CycleCost
    latency::Float64           # Cycles until result ready
    throughput::Float64        # Cycles between independent ops
end

Base.:+(c1::CycleCost, c2::CycleCost) = CycleCost(c1.latency + c2.latency, c1.throughput + c2.throughput)

const COST_TABLE = Dict{Tuple{SIMDOp, RegType}, CycleCost}(
    # Arithmetic - all have ~4 cycle latency, 0.5 cycle throughput (2 per cycle)
    (OP_ADD, REG_XMM) => CycleCost(4.0, 0.5),
    (OP_ADD, REG_YMM) => CycleCost(4.0, 0.5),
    (OP_ADD, REG_ZMM) => CycleCost(4.0, 0.5),
    (OP_SUB, REG_XMM) => CycleCost(4.0, 0.5),
    (OP_SUB, REG_YMM) => CycleCost(4.0, 0.5),
    (OP_SUB, REG_ZMM) => CycleCost(4.0, 0.5),
    (OP_MUL, REG_XMM) => CycleCost(4.0, 0.5),
    (OP_MUL, REG_YMM) => CycleCost(4.0, 0.5),
    (OP_MUL, REG_ZMM) => CycleCost(4.0, 0.5),

    # XOR is MUCH cheaper! 1 cycle latency, 3 per cycle throughput
    (OP_XOR, REG_XMM) => CycleCost(1.0, 0.33),
    (OP_XOR, REG_YMM) => CycleCost(1.0, 0.33),
    (OP_XOR, REG_ZMM) => CycleCost(1.0, 0.33),

    # Shuffles - within lane is cheap, cross-lane is expensive
    (OP_SHUFFLE, REG_XMM) => CycleCost(1.0, 1.0),   # Always within-lane
    (OP_SHUFFLE, REG_YMM) => CycleCost(3.0, 1.0),   # May cross lanes
    (OP_SHUFFLE, REG_ZMM) => CycleCost(3.0, 1.0),

    # Memory
    (OP_LOAD, REG_XMM) => CycleCost(5.0, 0.5),      # L1 cache latency
    (OP_LOAD, REG_YMM) => CycleCost(5.0, 0.5),
    (OP_LOAD, REG_ZMM) => CycleCost(5.0, 0.5),
    (OP_STORE, REG_XMM) => CycleCost(1.0, 1.0),
    (OP_STORE, REG_YMM) => CycleCost(1.0, 1.0),
    (OP_STORE, REG_ZMM) => CycleCost(1.0, 1.0),
)

get_cost(op::SIMDOp, reg::RegType) = get(COST_TABLE, (op, reg), CycleCost(10.0, 1.0))

# ============================================================================
# SIMD Expression Node
# ============================================================================

"""
Node in SIMD expression DAG
"""
mutable struct SIMDNode
    op::SIMDOp
    id::Int
    inputs::Vector{Union{SIMDNode, Symbol}}
    output::Symbol
    reg_type::RegType
    metadata::Dict{Symbol, Any}
    dependencies::Vector{Int}     # For dependency tracking
    level::Int                    # Recursion level
end

function SIMDNode(op::SIMDOp, inputs, output::Symbol;
                  reg_type=REG_XMM, metadata=Dict{Symbol,Any}(),
                  dependencies=Int[], level=0, id=-1)
    SIMDNode(op, id, inputs, output, reg_type, metadata, dependencies, level)
end

# ============================================================================
# Algebraic Rewrite Rules
# ============================================================================

"""
Key optimization: MUL by sign pattern → XOR

Example: v * [1, 1, -1, 1] becomes v ⊻ [0x0, 0x0, 0x80000000, 0x0]

This is 3-4x faster! (1 cycle vs 4 cycles latency)
"""
function rewrite_mul_to_xor!(node::SIMDNode)
    if node.op != OP_MUL
        return node
    end

    # Check if multiplying by a sign pattern (all ±1)
    if haskey(node.metadata, :sign_pattern)
        pattern = node.metadata[:sign_pattern]
        if all(x -> abs(x) ≈ 1.0, pattern)
            # Convert to XOR with sign bit mask
            # Sign bit for Float32: 0x80000000
            mask = [x < 0 ? 0x80000000 : 0x00000000 for x in pattern]

            node.op = OP_XOR
            node.metadata[:xor_mask] = mask
            delete!(node.metadata, :sign_pattern)

            @debug "Rewrote MUL to XOR" pattern mask savings_cycles=3.0
        end
    end

    return node
end

"""
Fuse consecutive shuffles: shuffle(shuffle(x, p1), p2) → shuffle(x, p1∘p2)
"""
function rewrite_fuse_shuffles!(node::SIMDNode)
    if node.op != OP_SHUFFLE || isempty(node.inputs)
        return node
    end

    input = node.inputs[1]
    if input isa SIMDNode && input.op == OP_SHUFFLE
        p1 = get(input.metadata, :pattern, nothing)
        p2 = get(node.metadata, :pattern, nothing)

        if !isnothing(p1) && !isnothing(p2) && length(p1) == length(p2)
            # Compose: (p1∘p2)[i] = p1[p2[i]]
            composed = [p1[p2[i]] for i in 1:length(p2)]
            node.inputs[1] = input.inputs[1]  # Skip intermediate shuffle
            node.metadata[:pattern] = composed

            @debug "Fused shuffles" p1 p2 composed
        end
    end

    return node
end

"""
Remove identity shuffles
"""
function rewrite_remove_identity_shuffle!(node::SIMDNode)
    if node.op == OP_SHUFFLE && haskey(node.metadata, :pattern)
        pattern = node.metadata[:pattern]
        if pattern == collect(1:length(pattern))
            # Identity - return input directly
            @debug "Removed identity shuffle" pattern
            # Mark as NOP (caller should bypass)
            node.op = OP_NOP
        end
    end
    return node
end

"""
Instruction scheduling: Reorder to avoid back-to-back dependent operations.

CPU can execute independent ops in parallel. If we have:
  op1 (produces A)
  op2 (uses A)        ← stall! waiting for op1
  op3 (independent)

Reorder to:
  op1 (produces A)
  op3 (independent)   ← executes while op1 finishes
  op2 (uses A)        ← no stall!
"""
function reorder_for_ilp!(nodes::Vector{SIMDNode})
    n = length(nodes)
    if n <= 2
        return nodes  # Too small to optimize
    end

    # Build dependency graph
    produces = Dict{Symbol, Int}()  # output symbol → node index
    for (i, node) in enumerate(nodes)
        produces[node.output] = i
    end

    reordered = SIMDNode[]
    available = Set(1:n)
    ready = Int[]  # Nodes with all dependencies satisfied

    # Find initially ready nodes (no dependencies)
    for i in 1:n
        deps_ready = true
        for inp in nodes[i].inputs
            if inp isa Symbol && haskey(produces, inp)
                deps_ready = false
                break
            end
        end
        if deps_ready
            push!(ready, i)
        end
    end

    while !isempty(available)
        if isempty(ready)
            # Deadlock or cycle - fallback to original order
            @warn "Dependency cycle detected in scheduling"
            return nodes
        end

        # Pick next ready node
        # Prefer nodes that unblock more operations
        idx = popfirst!(ready)
        push!(reordered, nodes[idx])
        delete!(available, idx)

        # Update ready list
        for i in available
            deps_ready = true
            for inp in nodes[i].inputs
                if inp isa Symbol && haskey(produces, inp)
                    dep_idx = produces[inp]
                    if dep_idx in available
                        deps_ready = false
                        break
                    end
                end
            end
            if deps_ready && !(i in ready)
                push!(ready, i)
            end
        end
    end

    @debug "Reordered $(length(nodes)) operations for ILP"
    return reordered
end

"""
Apply all algebraic rewrites to a node
"""
function apply_rewrites!(node::SIMDNode)
    rewrite_mul_to_xor!(node)
    rewrite_fuse_shuffles!(node)
    rewrite_remove_identity_shuffle!(node)
    return node
end

# ============================================================================
# Code Generation
# ============================================================================

"""
Generate Julia/SIMD.jl code from a SIMD node
"""
function codegen(node::SIMDNode, ::Type{T}=Float32; indent=0) where T
    ind = "    " ^ indent

    if node.op == OP_NOP
        return ""  # Skip NOP nodes
    end

    if node.op == OP_ADD
        lhs = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]
        rhs = node.inputs[2] isa SIMDNode ? node.inputs[2].output : node.inputs[2]
        return "$(ind)$(node.output) = $lhs + $rhs\n"
    end

    if node.op == OP_SUB
        lhs = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]
        rhs = node.inputs[2] isa SIMDNode ? node.inputs[2].output : node.inputs[2]
        return "$(ind)$(node.output) = $lhs - $rhs\n"
    end

    if node.op == OP_MUL
        lhs = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]
        rhs = node.inputs[2] isa SIMDNode ? node.inputs[2].output : node.inputs[2]
        return "$(ind)$(node.output) = $lhs * $rhs\n"
    end

    if node.op == OP_XOR
        vec = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]

        if haskey(node.metadata, :xor_mask)
            mask = node.metadata[:xor_mask]
            n = length(mask)
            mask_hex = ["0x" * string(m, base=16, pad=8) for m in mask]
            return """$(ind)$(node.output) = begin
                $(ind)_tmp_uint = reinterpret(Vec{$n,UInt32}, $vec)
$(ind)  _tmp_xor = _tmp_uint ⊻ Vec{$n,UInt32}(($(join(mask_hex, ", "))))
$(ind)    reinterpret(Vec{$n,$T}, _tmp_xor)
$(ind)end
"""
        end
    end

    if node.op == OP_SHUFFLE
        vec = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]

        if haskey(node.metadata, :pattern)
            pattern = node.metadata[:pattern]
            # SIMD.jl uses 0-based indexing for shufflevector
            pattern_0idx = tuple((p - 1 for p in pattern)...)
            return "$(ind)$(node.output) = shufflevector($vec, Val($pattern_0idx))\n"
        end
    end

    if node.op == OP_LOAD
        addr = get(node.metadata, :addr, :px)
        n_floats = get(node.metadata, :n_floats, 4)
        offset = get(node.metadata, :offset, 0)

        return """$(ind)$(node.output) = begin
$(ind)    LANE = VecRange{$n_floats}(0)
$(ind)    $addr[LANE + $(offset + 1)]
$(ind)end
"""
    end

    if node.op == OP_STORE
        vec = node.inputs[1] isa SIMDNode ? node.inputs[1].output : node.inputs[1]
        addr = get(node.metadata, :addr, :py)
        n_floats = get(node.metadata, :n_floats, 4)
        offset = get(node.metadata, :offset, 0)

        return """$(ind)begin
$(ind)    LANE = VecRange{$n_floats}(0)
$(ind)    $addr[LANE + $(offset + 1)] = $vec
$(ind)end
"""
    end

    return "$(ind)# Unknown op: $(node.op)\n"
end

# ============================================================================
# Integration: recfft2 → SIMD Expression DAG
# ============================================================================

"""
Operation DAG for managing SIMD nodes
"""
mutable struct OperationDAG
    ops::Vector{SIMDNode}
    counter::Int
    symbol_map::Dict{Symbol, Int}
end

OperationDAG() = OperationDAG(SIMDNode[], 0, Dict{Symbol,Int}())

function add_op!(dag::OperationDAG, node::SIMDNode)
    dag.counter += 1
    node.id = dag.counter
    push!(dag.ops, node)
    dag.symbol_map[node.output] = node.id
    return node.id
end

"""
Generate SIMD FFT kernel from recursive decomposition.

This is the KEY function - it builds the expression DAG from your recfft2 logic,
then applies algebraic rewrites, then generates code.
"""
function recfft2_simd_dag(n::Int, ::Type{T}; level=0) where T
    dag = OperationDAG()

    if n == 1
        # Identity - no ops
        return dag, [:x1]
    end

    if n == 2
        # Base case: 2-point butterfly
        # t1 = x1 + x2
        # t2 = x1 - x2

        add_id = add_op!(dag, SIMDNode(
            OP_ADD,
            [:x1, :x2],
            :t1,
            reg_type=REG_XMM,
            level=level
        ))

        sub_id = add_op!(dag, SIMDNode(
            OP_SUB,
            [:x1, :x2],
            :t2,
            reg_type=REG_XMM,
            level=level,
            dependencies=[add_id]  # Sub can execute in parallel, but mark for analysis
        ))

        return dag, [:t1, :t2]
    end

    # Recursive case
    n2 = n ÷ 2

    # Process even elements: FFT(x[0], x[2], x[4], ...)
    dag_even, t_even = recfft2_simd_dag(n2, T; level=level+1)
    for op in dag_even.ops
        add_op!(dag, op)
    end

    # Process odd elements: FFT(x[1], x[3], x[5], ...)
    dag_odd, t_odd = recfft2_simd_dag(n2, T; level=level+1)
    for op in dag_odd.ops
        add_op!(dag, op)
    end

    # Combine with butterflies and twiddle factors
    y = Symbol[]
    for i in 1:n2
        # Twiddle factor: W_n^(i-1) = e^(-2πi(i-1)/n)
        w = cispi(-2*(i-1)/n)

        # Decompose twiddle into sign pattern if possible
        # For n=4: w_0=1, w_1=-im → can optimize!
        is_sign_pattern = (real(w) == 0 || real(w) == 1 || real(w) == -1) &&
                         (imag(w) == 0 || imag(w) == 1 || imag(w) == -1)

        twiddle_sym = Symbol("t_tw_", i)

        if is_sign_pattern && abs(w) ≈ 1.0
            # Can optimize: w is just sign flips
            # e.g., -im = [0, -1] → shuffle + sign flip
            if w ≈ 1.0
                twiddle_sym = t_odd[i]  # No twiddle needed
            elseif w ≈ -im
                # -im rotation: (a+bi)*(-i) = b-ai
                # As vector: [a,b] → [b,-a]
                # Implementation: shuffle [b,a] then XOR to flip sign of second element

                # TODO: Full implementation would handle this
                # For now, use MUL and let rewrite handle it
                mul_id = add_op!(dag, SIMDNode(
                    OP_MUL,
                    [t_odd[i], :w_neg_im],
                    twiddle_sym,
                    reg_type=REG_XMM,
                    metadata=Dict(
                        :twiddle_factor => w,
                        :sign_pattern => [1.0, -1.0]  # Will trigger MUL→XOR rewrite
                    ),
                    level=level
                ))
            else
                # General sign pattern
                mul_id = add_op!(dag, SIMDNode(
                    OP_MUL,
                    [t_odd[i], Symbol("w_$i")],
                    twiddle_sym,
                    reg_type=REG_XMM,
                    metadata=Dict(
                        :twiddle_factor => w,
                        :sign_pattern => [real(w), imag(w)]
                    ),
                    level=level
                ))
            end
        else
            # General complex multiply (more expensive)
            mul_id = add_op!(dag, SIMDNode(
                OP_MUL,
                [t_odd[i], Symbol("w_$i")],
                twiddle_sym,
                reg_type=REG_XMM,
                metadata=Dict(:twiddle_factor => w),
                level=level
            ))
        end

        # Butterfly: y[i] = t_even[i] + twiddle * t_odd[i]
        #            y[i+n2] = t_even[i] - twiddle * t_odd[i]
        y_plus = Symbol("y", i)
        y_minus = Symbol("y", i+n2)

        add_op!(dag, SIMDNode(
            OP_ADD,
            [t_even[i], twiddle_sym],
            y_plus,
            reg_type=REG_XMM,
            level=level
        ))

        add_op!(dag, SIMDNode(
            OP_SUB,
            [t_even[i], twiddle_sym],
            y_minus,
            reg_type=REG_XMM,
            level=level
        ))

        push!(y, y_plus)
    end

    for i in 1:n2
        push!(y, Symbol("y", i+n2))
    end

    return dag, y
end

"""
Generate complete optimized SIMD kernel
"""
function generate_optimized_kernel(n::Int, ::Type{T}=Float32; name="vfft$(n)_opt") where T
    # Build expression DAG
    dag, outputs = recfft2_simd_dag(n, T)

    # Apply algebraic rewrites to each node
    for node in dag.ops
        apply_rewrites!(node)
    end

    # Reorder for instruction-level parallelism
    dag.ops = reorder_for_ilp!(dag.ops)

    # Generate code
    code_lines = String[]
    push!(code_lines, "@inline function $name(px::Vector{$T}, py::Vector{$T})")
    push!(code_lines, "    @inbounds @fastmath begin")

    for node in dag.ops
        code = codegen(node, T, indent=2)
        if !isempty(code)
            push!(code_lines, code)
        end
    end

    push!(code_lines, "    end")
    push!(code_lines, " end")

    return join(code_lines, ""), dag
end

# ============================================================================
# Exports
# ============================================================================

export SIMDOp, RegType, SIMDNode, OperationDAG
export get_cost, apply_rewrites!
export recfft2_simd_dag, generate_optimized_kernel
export codegen, add_op!
