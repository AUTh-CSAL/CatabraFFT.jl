module radix7_family

const Cp1, Sp1 = 0.6234898018587336, 0.7818314824680298
const Cp2, Sp2 = 0.22252093395631448, 0.9749279121818236
const Cp3, Sp3 = 0.9009688679024191, 0.4338837391175581

@inline function fft7_shell!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s

        a, b, c, d, e, f, g = x[q], x[q + s], x[q + 2s], x[q + 3s], x[q + 4s], x[q + 5s], x[q + 6s]

        bpg, cpf, dpe = b + g, c + f, d + e
        bmg, cmf, dme = b - g, c - f, d - e
        x1, y1 = a + bpg*Cp1 - cpf*Cp2 - dpe*Cp3, im*(-bmg*Sp1 - cmf*Sp2 - dme*Sp3)
        x2, y2 = a - bpg*Cp2 - cpf*Cp3 + dpe*Cp1, im*(-bmg*Sp2 + cmf*Sp3 + dme*Sp1)
        x3, y3 = a - bpg*Cp3 + cpf*Cp1 - dpe*Cp2, im*(-bmg*Sp3 + cmf*Sp1 - dme*Sp2)

        y[q] = a + bpg + cpf + dpe
        y[q + s] = x1 + y1
        y[q + 2s] = x2 + y2
        y[q + 3s] = x3 + y3
        y[q + 4s] = x3 - y3
        y[q + 5s] = x2 - y2
        y[q + 6s] = x1 - y1
    end
end

@inline function fft7_shell_y!(y::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s

        a, b, c, d, e, f, g = y[q], y[q + s], y[q + 2s], y[q + 3s], y[q + 4s], y[q + 5s], y[q + 6s]

        bpg, cpf, dpe = b + g, c + f, d + e
        bmg, cmf, dme = b - g, c - f, d - e
        x1, y1 = a + bpg*Cp1 - cpf*Cp2 - dpe*Cp3, im*(-bmg*Sp1 - cmf*Sp2 - dme*Sp3)
        x2, y2 = a - bpg*Cp2 - cpf*Cp3 + dpe*Cp1, im*(-bmg*Sp2 + cmf*Sp3 + dme*Sp1)
        x3, y3 = a - bpg*Cp3 + cpf*Cp1 - dpe*Cp2, im*(-bmg*Sp3 + cmf*Sp1 - dme*Sp2)

        y[q] = a + bpg + cpf + dpe
        y[q + s] = x1 + y1
        y[q + 2s] = x2 + y2
        y[q + 3s] = x3 + y3
        y[q + 4s] = x3 - y3
        y[q + 5s] = x2 - y2
        y[q + 6s] = x1 - y1
    end
end

@inline function fft7_shell_layered!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int, n1::Int, theta::Float64)
    @inbounds @simd for p in 0:(n1 - 1)
            w1p = cispi(-p * theta)
            w2p = w1p * w1p
            w3p = w2p * w1p
            w4p = w3p * w1p
            w5p = w4p * w1p
            w6p = w5p * w1p
            @inbounds @simd for q in 1:s

            a, b, c, d, e, f, g = x[q + s*p], x[q + s*(p + n1)], x[q + s*(p + 2n1)], x[q + s*(p + 3n1)], x[q + s*(p + 4n1)], x[q + s*(p + 5n1)], x[q + s*(p + 6n1)]

            bpg, cpf, dpe = b + g, c + f, d + e
            bmg, cmf, dme = b - g, c - f, d - e
            x1, y1 = a + bpg*Cp1 - cpf*Cp2 - dpe*Cp3, im*(-bmg*Sp1 - cmf*Sp2 - dme*Sp3)
            x2, y2 = a - bpg*Cp2 - cpf*Cp3 + dpe*Cp1, im*(-bmg*Sp2 + cmf*Sp3 + dme*Sp1)
            x3, y3 = a - bpg*Cp3 + cpf*Cp1 - dpe*Cp2, im*(-bmg*Sp3 + cmf*Sp1 - dme*Sp2)
            
            y[q + s*7p] = a + bpg + cpf + dpe
            y[q + s*(7p + 1)] = w1p * (x1 + y1)
            y[q + s*(7p + 2)] = w2p * (x2 + y2)
            y[q + s*(7p + 3)] = w3p * (x3 + y3)
            y[q + s*(7p + 4)] = w4p * (x3 - y3)
            y[q + s*(7p + 5)] = w5p * (x2 - y2)
            y[q + s*(7p + 6)] = w6p * (x1 - y1)
            end
        end
end

end