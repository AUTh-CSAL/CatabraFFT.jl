#include("radix_plan.jl")

using .Radix_Plan

const FLAG = Int8

const NO_FLAG = FLAG(0)
const MEASURE = FLAG(1)
const ENCHANT = FLAG(2)

"""
Represents a universal FFT computation strategy that can encode any FFT pattern:
- Pure radix-a^s for a ∈ {2^a,3^b,5^c,7^d}
- Mixed-radix decompositions
- Prime-size FFTs using Rader's algorithm (TODO Bluestein sometimes)
- Trivial cases ( n=1 )
- TODO Dynamic Radix Generation of radix-m, m < 16
"""
struct Spell{T} <: AbstractFFTs.Plan{T}
    # Core properties
    n::Int                    # Transform size
    datatype::Type{T}        # Numeric type (Float16/Float32/Float64)
    
    # Optimization flags
    flag::FLAG

    # Inverse transform reference
    inverse::Union{Nothing, Spell{T}}  # Reference to inverse transform plan
end

function Spell(n::Int, ::Type{T}, flag::FLAG) where T <: AbstractFloat
    Spell{T}(
        n,
        T,
        flag,
        nothing
    )
end

# EXPANDED SPELL SECTION
#=
struct Spell{T} <: AbstractFFTs.Plan{T}
    # Core properties
    n::Int                    # Transform size
    strategy::Symbol          # :radix, :mixed, :prime, or :trivial
    datatype::Type{T}        # Numeric type (Float16/Float32/Float64)
    
    # Strategy-specific parameters
    parameters::Dict{Symbol, Any}  # Holds strategy-specific parameters
    
    # Computation components
    components::Vector{Any}   # Holds sub-components (RadixPlan, MixedRadixFFT, etc.)
    
    # Optimization flags
    flags::Dict{Symbol, Bool} # Various optimization flags
    
    # Cache and preprocessing data
    cache::Dict{Symbol, Any}  # Precomputed data (twiddles, permutations, etc.)
    
    # Inverse transform reference
    inverse::Union{Nothing, Spell{T}}  # Reference to inverse transform plan
end

"""
Constructor for pure radix-a^s FFTs
"""
function Spell(n::Int, radix::Int, ::Type{T}) where T <: AbstractFloat
    @assert is_power_of(n, radix) "n must be a power of $radix"
    
    parameters = Dict{Symbol, Any}(
        :radix => radix,
        :stages => Int(log(radix, n))
    )
    
    #components = create_radix_components(n, radix, T)
    
    flags = Dict{Symbol, Bool}(
        :inplace => true,
        :threaded => Threads.nthreads() > 1,
        :vectorized => true
    )
    
    cache = Dict{Symbol, Any}(
        :twiddles => precompute_twiddles(n, T),
        :bit_reversal => precompute_bit_reversal(n)
    )
    
    Spell{T}(
        n,
        :radix,
        T,
        parameters,
        components,
        flags,
        cache,
        nothing
    )
end

"""
Constructor for mixed-radix FFTs
"""
function Spell(p::Int, m::Int, ::Type{T}) where T <: AbstractFloat
    n = p * m
    
    parameters = Dict{Symbol, Any}(
        :p => p,
        :m => m
    )
    
    components = [
        create_mixed_radix_components(p, m, T)
    ]
    
    flags = Dict{Symbol, Bool}(
        :inplace => true,
        :threaded => Threads.nthreads() > 1,
        :vectorized => true
    )
    
    cache = Dict{Symbol, Any}(
        :twiddles => precompute_mixed_twiddles(p, m, T),
        :permutation => precompute_mixed_permutation(p, m)
    )
    
    Spell{T}(
        n,
        :mixed,
        T,
        parameters,
        components,
        flags,
        cache,
        nothing
    )
end

"""
Constructor for prime-size FFTs using Rader's algorithm
"""
function Spell(n::Int, ::Type{T}, ::Val{:prime}) where T <: AbstractFloat
    @assert isprime(n) "n must be prime"
    
    g = find_primitive_root(n)
    
    parameters = Dict{Symbol, Any}(
        :primitive_root => g,
        :conv_size => n - 1
    )
    
    # Create component for the cyclic convolution
    conv_spell = create_convolution_spell(n - 1, T)
    components = [conv_spell]
    
    flags = Dict{Symbol, Bool}(
        :inplace => true,
        :threaded => false,  # Prime-size typically too small for threading
        :vectorized => true
    )
    
    cache = Dict{Symbol, Any}(
        :powers => precompute_powers(g, n),
        :twiddles => precompute_rader_twiddles(n, T)
    )
    
    Spell{T}(
        n,
        :prime,
        T,
        parameters,
        components,
        flags,
        cache,
        nothing
    )
end

"""
Encodes a Spell into a format suitable for serialization
"""
function encode(spell::Spell{T}) where T
    Dict{String, Any}(
        "version" => "1.0",
        "size" => spell.n,
        "strategy" => String(spell.strategy),
        "datatype" => string(T),
        "parameters" => spell.parameters,
        "flags" => spell.flags,
        "cache_keys" => collect(keys(spell.cache)),
        "components" => map(encode_component, spell.components)
    )
end

"""
Decodes a serialized format back into a Spell
"""
function decode(data::Dict{String, Any}, ::Type{T}) where T
    n = data["size"]
    strategy = Symbol(data["strategy"])
    
    # Reconstruct based on strategy
    spell = if strategy == :radix
        Spell(n, data["parameters"][:radix], T)
    elseif strategy == :mixed
        p, m = data["parameters"][:p], data["parameters"][:m]
        Spell(p, m, T)
    elseif strategy == :prime
        Spell(n, T, Val(:prime))
    else  # :trivial
        Spell{T}(n, strategy, T, Dict(), [], Dict(), Dict(), nothing)
    end
    
    # Update flags if they differ
    merge!(spell.flags, data["flags"])
    
    spell
end

# Helper functions for component encoding/decoding
function encode_component(comp::RadixPlan)
    Dict{String, Any}(
        "type" => "radix",
        "stride" => comp.stride,
        "count" => comp.count
    )
end

function encode_component(comp::MixedRadixFFT)
    Dict{String, Any}(
        "type" => "mixed",
        "p" => comp.p,
        "m" => comp.m
    )
end

# Utility functions for twiddle factors and bit reversal
function precompute_twiddles(n::Int, ::Type{T}) where T
    [exp(-2π*im*k/n) for k in 0:n-1]
end

function precompute_bit_reversal(n::Int)
    bits = Int(log2(n))
    [reverse_bits(i-1, bits) + 1 for i in 1:n]
end

function reverse_bits(x::Int, bits::Int)
    result = 0
    for i in 0:bits-1
        result = (result << 1) | ((x >> i) & 1)
    end
    result
end

struct SpellComponent
    type::String    # Component type (radix, mixed, prime)
    params::Dict{String, Any}
    children::Vector{SpellComponent}
end

struct FFTSpellEncoding
    version::String  # Schema version
    size::Int       # Input size n
    strategy::String # Primary decomposition strategy
    components::Vector{SpellComponent}
    metadata::Dict{String, Any}
end


# Strategy Encodings:
# RADIX-<base> : For pure radix-a^s computations
# MIXED-<p>-<m> : For mixed-radix decompositions
# PRIME-RADER   : For prime-size FFTs using Rader's algorithm
# TRIVIAL       : For n=1 case

function encode_spell(spell::Spell{T}) where T
    n = spell.n
    
    # Determine base strategy
    if n == 1
        return FFTSpellEncoding(
            "1.0",
            1,
            "TRIVIAL",
            SpellComponent[],
            Dict("datatype" => string(T))
        )
    elseif !isnothing(spell.radixplan)
        # Handle pure radix cases
        base = determine_radix_base(spell.radixplan)
        return FFTSpellEncoding(
            "1.0",
            n,
            "RADIX-$base",
            [encode_radix_plan(plan) for plan in spell.radixplan],
            Dict("datatype" => string(T))
        )
    elseif !isnothing(spell.mixed)
        # Handle mixed-radix case
        p, m = get_mixed_radix_factors(spell.mixed)
        return FFTSpellEncoding(
            "1.0",
            n,
            "MIXED-$p-$m",
            [encode_mixed_radix(spell.mixed)],
            Dict("datatype" => string(T))
        )
    else
        # Prime case using Rader's algorithm
        return FFTSpellEncoding(
            "1.0",
            n,
            "PRIME-RADER",
            [encode_prime_rader(n)],
            Dict("datatype" => string(T))
        )
    end
end

function decode_spell(encoding::FFTSpellEncoding, ::Type{T}) where T
    n = encoding.size
    
    # Reconstruct the appropriate spell based on strategy
    spell = if encoding.strategy == "TRIVIAL"
        Spell{T}(1, nothing, nothing, false, nothing)
    elseif startswith(encoding.strategy, "RADIX")
        base = parse(Int, split(encoding.strategy, "-")[2])
        radixplan = [decode_radix_component(comp) for comp in encoding.components]
        Spell{T}(n, nothing, radixplan, false, nothing)
    elseif startswith(encoding.strategy, "MIXED")
        p, m = parse.(Int, split(encoding.strategy, "-")[2:3])
        mixed = decode_mixed_component(encoding.components[1], p, m, T)
        Spell{T}(n, mixed, nothing, false, nothing)
    else # PRIME-RADER
        prime_plan = decode_prime_component(encoding.components[1], n, T)
        Spell{T}(n, nothing, nothing, false, nothing)
    end
    
    return spell
end

# Helper functions for component encoding/decoding
function encode_radix_plan(plan::RadixPlan)
    SpellComponent(
        "radix",
        Dict(
            "stride" => plan.stride,
            "count" => plan.count,
            "base" => plan.base
        ),
        SpellComponent[]
    )
end

function encode_mixed_radix(mixed::MixedRadixFFT)
    SpellComponent(
        "mixed",
        Dict(
            "p" => mixed.p,
            "m" => mixed.m,
            "twiddles" => collect(mixed.twiddles)
        ),
        SpellComponent[]
    )
end

function encode_prime_rader(n::Int)
    SpellComponent(
        "prime",
        Dict(
            "size" => n,
            "generator" => find_primitive_root(n)
        ),
        SpellComponent[]
    )
end

=#