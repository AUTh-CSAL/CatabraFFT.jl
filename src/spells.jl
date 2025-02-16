
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
struct Spell{T <: AbstractFloat}
    size::Tuple{Int,}
    region::Vector{Int}
    pinv::Base.RefValue{Spell}
    flag::FLAG
    fft_func::Function  # Direct storage of generated function
    # Optional: Add these fields for fast lookups
    n::Int
    type::Type{T}
    
    function Spell{T}(n::Int, flag::FLAG, func::Function) where T
        new{T}((n,), [1], Ref{Spell}(), flag, func, n, T)
    end
end

