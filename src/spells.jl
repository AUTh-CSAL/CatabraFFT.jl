using .Radix_Plan

const FLAG = Int8
const NO_FLAG = FLAG(0)
const MEASURE = FLAG(1)
const ENCHANT = FLAG(2)

# Type-encoded Spell that contains the entire plan in its type
struct Spell{T<:AbstractFloat, N, DECOMP, FLAG_VAL}
    # Minimal runtime data
    size::NTuple{1,Int}
    
    function Spell{T,N,DECOMP,FLAG_VAL}() where {T,N,DECOMP,FLAG_VAL}
        new{T,N,DECOMP,FLAG_VAL}((N,))
    end
end

# Convert RadixPlan to type-encoded decomposition tuple
function plan_to_decomp(plan::RadixPlan)
    decomp = []
    for op in plan.operations
        radix = get_radix_divisor(op.op_type)
        push!(decomp, (radix=radix, stride=op.stride, n_groups=op.n_groups))
    end
    return Tuple(decomp)
end

# Create Spell from RadixPlan (after benchmarking)
function RadixPlan_to_Spell(plan::RadixPlan{T}, flag::FLAG=NO_FLAG) where T
    decomp = plan_to_decomp(plan)
    return Spell{T, plan.n, decomp, Int(flag)}()
end

# Direct constructor for known decomposition
function Spell(n::Int, ::Type{T}, decomp::Tuple, flag::FLAG=NO_FLAG) where T
    Spell{T, n, decomp, Int(flag)}()
end