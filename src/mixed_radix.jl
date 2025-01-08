using LoopVectorization

include("spells.jl")

if !@isdefined(MixedRadixFFT)
struct MixedRadixFFT{T <: AbstractFloat}
    p::Int
    m::Int
    W::Matrix{Complex{T}}
    Fm::Function  # Row-wise FFT function
    Fp::Function  # Column-wise FFT function
end
end

if !@isdefined(mixed_radix_cache)
    const mixed_radix_cache = Dict{Tuple{Int, Int, DataType}, MixedRadixFFT}() 
end

function MixedRadixFFT(p::Int, m::Int, ::Type{T}, flag::FLAG)::MixedRadixFFT where {T<:AbstractFloat}
    key = (p, m, T)
    haskey(mixed_radix_cache, key) && return mixed_radix_cache[key]

    W = D(p, m, T)
    Fm = recursive_F(m, T, flag)
    Fp = recursive_F(p, T, flag)
    mixed_radix_cache[key] = MixedRadixFFT{T}(p, m, W, Fm, Fp)
    MixedRadixFFT{T}(p, m, W, Fm, Fp)
end

@inline function D(p,m, ::Type{T})::Matrix where {T<:AbstractFloat}
  w = cispi.(T(-2/(p*m)) * collect(1:m-1))
  d = zeros(Complex{T},(p-1)*(m-1))

  @inbounds d[1:m-1] .= w

  @inbounds @simd for j in 2:p-1
        @views d[(j-1)*(m-1)+1:j*(m-1)] .= w .* view(d, (j-2)*(m-1)+1:(j-1)*(m-1))
  end

  return reshape(d, m-1, p-1)
end

# vec(transpose(F(m)*(W.*(Xmp*F(p))))) ≈ y
function generate_formulation_fft(plan::MixedRadixFFT, ::Type{T})::Function where {T<:AbstractFloat}
    p, m = plan.p, plan.m
    W = D(p,m, T)

    Ypm = Matrix{Complex{T}}(undef, p, m)
    Xpm = Matrix{Complex{T}}(undef, p, m)

    return @inline function (y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}})

        Xpm .= reshape(x, p, m)
        Ypm .= zero(Complex{T})

        @inbounds @simd for i in 1:p
            plan.Fm(@view(Ypm[i, :]), @view(Xpm[i, :]))
        end

        # TODO FIND FASTER FFT-LIKE POINT-WISE MULTIPLICATION THAN POINT-WISE O(p*m) W MATRIX MAT MUL
        @inbounds @simd for i in 1:p-1
            @inbounds @simd for j in 1:m-1
                Ypm[i + 1, j + 1] *= W[j,i]
            end
        end

        # Step 3: Transpose the result and apply Fp column-wise
        # (W^T (X_pm Fm))^T Fp
        @inbounds @simd for j in 1:m
            plan.Fp(@view(Xpm[:, j]), @view(Ypm[:, j]))
        end

        # Step 4: Write the output to the vector y
        y .= vec(transpose(Xpm))

        return nothing
    end
end
