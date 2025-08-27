# spells.jl
# This file should be included after radix_plan.jl so RadixPlan is available

const FLAG = Int8
const NO_FLAG = FLAG(0)
const MEASURE = FLAG(1)
const ENCHANT = FLAG(2)

# Pure type-level Spell - truly no runtime data, everything encoded in type parameters
# This serves as both the plan and the cache key
struct Spell{T <: AbstractFloat, N, DECOMP, FLAG_VAL}
    # Add minimal fields for AbstractFFTs compatibility only
    size::Tuple{Int}
    region::UnitRange{Int}
    pinv::Base.RefValue{Any}
    
    # Internal constructor to set default values
    function Spell{T, N, DECOMP, FLAG_VAL}() where {T, N, DECOMP, FLAG_VAL}
        new{T, N, DECOMP, FLAG_VAL}((N,), 1:1, Ref{Any}())
    end
end

# Convenience constructors for different scenarios
function Spell(::Type{T}, n::Int, decomp::Tuple=(), flag::FLAG=NO_FLAG) where T<:AbstractFloat
    Spell{T, n, typeof(decomp), Int(flag)}()
end

# Constructor without type parameter for decomp
function Spell(::Type{T}, n::Int, flag::FLAG) where T<:AbstractFloat
    Spell{T, n, Tuple{}, Int(flag)}()
end

# Convert RadixPlan to type-encoded Spell
function RadixPlan_to_Spell(plan::Radix_Plan.RadixPlan, ::Type{T}, flag::FLAG=NO_FLAG) where T<:AbstractFloat
    decomp = plan_to_decomp(plan)
    Spell{T, plan.n, typeof(decomp), Int(flag)}()
end

function plan_to_decomp(plan::Radix_Plan.RadixPlan)
    decomp = []
    for op in plan.operations
        radix = get_radix_divisor(op.op_type)
        push!(decomp, (radix=radix, stride=op.stride, n_groups=op.n_groups))
    end
    return Tuple(decomp)
end

# Type-level cache key generation for maximum performance
@inline function spell_type(::Type{T}, n::Int, decomp::Tuple, flag::FLAG) where T<:AbstractFloat
    return Spell{T, n, typeof(decomp), Int(flag)}
end

@inline function spell_type(::Type{T}, n::Int, flag::FLAG) where T<:AbstractFloat
    return Spell{T, n, Tuple{}, Int(flag)}
end