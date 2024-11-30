module radix5_family

const Cp1, Sp1 = 0.8090169943749473, 0.5877852522924731
const Cp2, Sp2 = 0.30901699437494734, 0.9510565162951536

using LoopVectorization, SIMD

@inline function fft5_shell!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c, d, e = x[q], x[q + s], x[q + 2s], x[q + 3s], x[q + 4s]

        bpe, dpc = b + e, d + c
        bme, dmc = b - e, d - c
        x1, y1 = a + bpe*Cp2 - dpc*Cp1, im*(-bme*Sp2 + dmc*Sp1)
        x2, y2 = a - bpe*Cp1 + dpc*Cp2, im*(bme*Sp1 + dmc*Sp2)
        
        y[q] = a + bpe + dpc
        y[q + s] = x1 + y1
        y[q + 2s] = x2 - y2
        y[q + 3s] = x2 + y2
        y[q + 4s] = x1 - y1
    end
end

@inline function fft5_shell_y!(y::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c, d, e = y[q], y[q + s], y[q + 2s], y[q + 3s], y[q + 4s]

        bpe, dpc = b + e, d + c
        bme, dmc = b - e, d - c
        x1, y1 = a + bpe*Cp2 - dpc*Cp1, im*(-bme*Sp2 + dmc*Sp1)
        x2, y2 = a - bpe*Cp1 + dpc*Cp2, im*(bme*Sp1 + dmc*Sp2)
        
        y[q] = a + bpe + dpc
        y[q + s] = x1 + y1
        y[q + 2s] = x2 - y2
        y[q + 3s] = x2 + y2
        y[q + 4s] = x1 - y1
    end
end

@inline function fft5_shell_layered!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int, n1::Int, theta::Float64)
    @inbounds @simd for p in 0:(n1 - 1)
        w1p = cispi(-p * theta)
        w2p = w1p * w1p
        w3p = w2p * w1p
        w4p = w3p * w1p
        @inbounds @simd for q in 1:s 
            a, b, c, d, e = x[q + s*p], x[q + s*(p + n1)], x[q + s*(p + 2n1)], x[q + s*(p + 3n1)], x[q + s*(p + 4n1)]

            bpe, dpc = b + e, d + c
            bme, dmc = b - e, d - c
            x1, y1 = a + bpe*Cp2 - dpc*Cp1, im*(-bme*Sp2 + dmc*Sp1)
            x2, y2 = a - bpe*Cp1 + dpc*Cp2, im*(bme*Sp1 + dmc*Sp2)

            y[q + s*5p] = a + bpe + dpc
            y[q + s*(5p + 1)] = w1p * (x1 + y1)
            y[q + s*(5p + 2)] = w2p * (x2 - y2)
            y[q + s*(5p + 3)] = w3p * (x2 + y2)
            y[q + s*(5p + 4)] = w4p * (x1 - y1)
        end
    end
end

end