module radix3_family

const Cp = 0.5
const Sp = 0.8660254037844386

const Cp_2_9, Sp_2_9 = 0.766044443118978, 0.6427876096865393
const Cp_4_9, Sp_4_9 = 0.17364817766693041, 0.984807753012208
const Cp_8_9, Sp_8_9 = 0.9396926207859083, 0.3420201433256689

using LoopVectorization, SIMD

@inline function fft9_shell!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c, d, e, f, g, h, i = x[q], x[q + s], x[q + 2s], x[q + 3s], x[q + 4s], x[q + 5s], x[q + 6s], x[q + 7s], x[q + 8s]
        
        t1, t2, t3, t4 = b + i, c + h, d + g, e + f
        m1, m2, m3, m4 = i - b, h - c, g - d, f - e
        t3_Cp, m3_Sp = t3 * Cp, m3 * Sp

        t11, t12 = t1 + t2 + t4, a + t3
        
        x1, y1 = a + Cp_2_9*t1 + Cp_4_9*t2 - t3_Cp - Cp_8_9*t4, im*(Sp_2_9*m1 + Sp_4_9*m2 + m3_Sp + Sp_8_9*m4)
        x2, y2 = a + Cp_4_9 * t1 - Cp_8_9 * t2 - t3_Cp + Cp_2_9 * t4, im * (Sp_4_9 * m1 + Sp_8_9 * m2 - m3_Sp - Sp_2_9 * m4) # WRONG
        
        x3, y3 = (-Cp*t11 + t12), im*Sp*(m1 - m2 + m4)
        x4, y4 = a - Cp_8_9 * t1 + Cp_2_9 * t2 - t3_Cp + Cp_4_9 * t4, im * (Sp_8_9 * m1 - Sp_2_9 * m2 + m3_Sp - Sp_4_9 * m4)
        
        y[q] = t11 + t12
        y[q + s] = x1 + y1
        y[q + 2s] = x2 + y2
        y[q + 3s] = x3 + y3
        y[q + 4s] = x4 + y4
        y[q + 5s] = x4 - y4
        y[q + 6s] = x3 - y3
        y[q + 7s] = x2 - y2
        y[q + 8s] = x1 - y1
    end
end

@inline function fft9_shell_y!(y::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c, d, e, f, g, h, i = y[q], y[q + s], y[q + 2s], y[q + 3s], y[q + 4s], y[q + 5s], y[q + 6s], y[q + 7s], y[q + 8s]
        
        t1, t2, t3, t4 = b + i, c + h, d + g, e + f
        m1, m2, m3, m4 = i - b, h - c, g - d, f - e
        t3_Cp, m3_Sp = t3 * Cp, m3 * Sp

        t11, t12 = t1 + t2 + t4, a + t3
        
        x1, y1 = a + Cp_2_9*t1 + Cp_4_9*t2 - t3_Cp - Cp_8_9*t4, im*(Sp_2_9*m1 + Sp_4_9*m2 + m3_Sp + Sp_8_9*m4)
        x2, y2 = a + Cp_4_9 * t1 - Cp_8_9 * t2 - t3_Cp + Cp_2_9 * t4, im * (Sp_4_9 * m1 + Sp_8_9 * m2 - m3_Sp - Sp_2_9 * m4) # WRONG
        
        x3, y3 = (-Cp*t11 + t12), im*Sp*(m1 - m2 + m4)
        x4, y4 = a - Cp_8_9 * t1 + Cp_2_9 * t2 - t3_Cp + Cp_4_9 * t4, im * (Sp_8_9 * m1 - Sp_2_9 * m2 + m3_Sp - Sp_4_9 * m4)
        
        y[q] = t11 + t12
        y[q + s] = x1 + y1
        y[q + 2s] = x2 + y2
        y[q + 3s] = x3 + y3
        y[q + 4s] = x4 + y4
        y[q + 5s] = x4 - y4
        y[q + 6s] = x3 - y3
        y[q + 7s] = x2 - y2
        y[q + 8s] = x1 - y1
    end
end

@inline function fft9_shell_layered!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int, n1::Int, theta::Float64)

    @inbounds @simd for q in 1:s
        a, b, c, d, e, f, g, h, i = x[q], x[q + s*n1], x[q + 2s*n1], x[q + 3s*n1], x[q + 4s*n1], x[q + 5s*n1], x[q + 6s*n1], x[q + 7s*n1], x[q + 8s*n1]
        
        t1, t2, t3, t4 = b + i, c + h, d + g, e + f
        m1, m2, m3, m4 = i - b, h - c, g - d, f - e
        t3_Cp, m3_Sp = t3 * Cp, m3 * Sp

        t11, t12 = t1 + t2 + t4, a + t3
        
        x1, y1 = a + Cp_2_9*t1 + Cp_4_9*t2 - t3_Cp - Cp_8_9*t4, im*(Sp_2_9*m1 + Sp_4_9*m2 + m3_Sp + Sp_8_9*m4)
        x2, y2 = a + Cp_4_9 * t1 - Cp_8_9 * t2 - t3_Cp + Cp_2_9 * t4, im * (Sp_4_9 * m1 + Sp_8_9 * m2 - m3_Sp - Sp_2_9 * m4) # WRONG
        
        x3, y3 = (-Cp*t11 + t12), im*Sp*(m1 - m2 + m4)
        x4, y4 = a - Cp_8_9 * t1 + Cp_2_9 * t2 - t3_Cp + Cp_4_9 * t4, im * (Sp_8_9 * m1 - Sp_2_9 * m2 + m3_Sp - Sp_4_9 * m4)
        
        y[q] = t11 + t12
        y[q + s] = x1 + y1
        y[q + 2s] = x2 + y2
        y[q + 3s] = x3 + y3
        y[q + 4s] = x4 + y4
        y[q + 5s] = x4 - y4
        y[q + 6s] = x3 - y3
        y[q + 7s] = x2 - y2
        y[q + 8s] = x1 - y1
    end

    @inbounds @simd for p in 1:(n1-1)
        w1p = cispi(-p*theta)
        w2p = w1p * w1p
        w3p = w2p * w1p
        w4p = w2p * w2p
        w5p = w3p * w2p
        w6p = w3p * w3p
        w7p = w4p * w3p
        w8p = w4p * w4p
    @inbounds @simd for q in 1:s
        a, b, c, d, e, f, g, h, i = x[q + s*p], x[q + s*(p + n1)], x[q + s*(p + 2n1)], x[q + s*(p + 3n1)], x[q + s*(p + 4n1)], x[q + s*(p + 5n1)], x[q + s*(p + 6n1)], x[q + s*(p + 7n1)], x[q + s*(p + 8n1)]
        
        t1, t2, t3, t4 = b + i, c + h, d + g, e + f
        m1, m2, m3, m4 = i - b, h - c, g - d, f - e
        t3_Cp, m3_Sp = t3 * Cp, m3 * Sp

        t11, t12 = t1 + t2 + t4, a + t3
        
        x1, y1 = a + Cp_2_9*t1 + Cp_4_9*t2 - t3_Cp - Cp_8_9*t4, im*(Sp_2_9*m1 + Sp_4_9*m2 + m3_Sp + Sp_8_9*m4)
        x2, y2 = a + Cp_4_9 * t1 - Cp_8_9 * t2 - t3_Cp + Cp_2_9 * t4, im * (Sp_4_9 * m1 + Sp_8_9 * m2 - m3_Sp - Sp_2_9 * m4) # WRONG
        
        x3, y3 = (-Cp*t11 + t12), im*Sp*(m1 - m2 + m4)
        x4, y4 = a - Cp_8_9 * t1 + Cp_2_9 * t2 - t3_Cp + Cp_4_9 * t4, im * (Sp_8_9 * m1 - Sp_2_9 * m2 + m3_Sp - Sp_4_9 * m4)
        
        y[q + s*9p] = t11 + t12
        y[q + s*(9p + 1)] = w1p*(x1 + y1)
        y[q + s*(9p + 2)] = w2p*(x2 + y2)
        y[q + s*(9p + 3)] = w3p*(x3 + y3)
        y[q + s*(9p + 4)] = w4p*(x4 + y4)
        y[q + s*(9p + 5)] = w5p*(x4 - y4)
        y[q + s*(9p + 6)] = w6p*(x3 - y3)
        y[q + s*(9p + 7)] = w7p*(x2 - y2)
        y[q + s*(9p + 8)] = w8p*(x1 - y1)
    end
    end
end

@inline function fft3_shell!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c = x[q], x[q + s], x[q + 2s]
        
        bpc = b + c
        jbmcS = im*(b - c)*Sp
        ambpcCp = a - bpc*Cp

        y[q] = a + bpc
        y[q + s] = ambpcCp - jbmcS
        y[q + 2s] = ambpcCp + jbmcS
    end
end

@inline function fft3_shell_y!(y::AbstractVector{ComplexF64}, s::Int)
    @inbounds @simd for q in 1:s
        a, b, c = y[q], y[q + s], y[q + 2s]
        
        bpc = b + c
        jbmcS = im*(b - c)*Sp
        ambpcCp = a - bpc*Cp

        y[q] = a + bpc
        y[q + s] = ambpcCp - jbmcS
        y[q + 2s] = ambpcCp + jbmcS
    end
end

@inline function fft3_shell_layered!(y::AbstractVector{ComplexF64}, x::AbstractVector{ComplexF64}, s::Int, n1::Int, theta::Float64)
    @inbounds @simd for p in 0:(n1-1)
        w1p = cispi(- p * theta)
        w2p = w1p^2
        @inbounds @simd for q in 1:s
            a, b, c = x[q + s * p], x[q + s * (p + n1)], x[q + s * (p + 2n1)]

            bpc = b + c
            jbmcS = im*(b - c)*Sp
            ambpcCp = a - bpc*Cp

            y[q + s * (3p + 0)] = a + bpc
            y[q + s * (3p + 1)] = w1p*(ambpcCp - jbmcS)
            y[q + s * (3p + 2)] = w2p*(ambpcCp + jbmcS)
        end
    end
end



@inline function fft3_codelet!(x_ptr::Ptr{Float64}, y_ptr::Ptr{Float64}, q::Int, s::Int)
    idx = (q - 1) * 16
    
    x0 = vload(Vec{2, Float64}, x_ptr + idx)
    x1 = vload(Vec{2, Float64}, x_ptr + idx + s * 16)
    x2 = vload(Vec{2, Float64}, x_ptr + idx + 2s * 16)

    #=
    println("x[q]: $x0, $(idx ÷ 16) ")
    println("x[q + s]: $x1, $((idx + s) ÷ 16)")
    println("x[q + 2s]: $x2, $((idx + 2s) ÷ 16)")
    =#

    t1 = x1 + x2
    x1 = x2 - x1
    x1 = Sp * Vec(x1[2], -x1[1])
    x2 = x0 - t1*Cp

    vstore(x0 + t1, y_ptr + idx)
    vstore(x2 - x1, y_ptr + idx + s * 16)
    vstore(x2 + x1, y_ptr + idx + 2s * 16)
end

@inline function fft3_codelet_y!(y_ptr::Ptr{Float64}, q::Int, s::Int)
    idx = (q - 1) * 16
    
    x0 = vload(Vec{2, Float64}, y_ptr + idx)
    x1 = vload(Vec{2, Float64}, y_ptr + idx + s * 16)
    x2 = vload(Vec{2, Float64}, y_ptr + idx + 2s * 16)

    t1 = x1 + x2
    x1 = x2 - x1
    x1 = Sp * Vec(x1[2], -x1[1])
    x2 = x0 - t1*Cp
    
    vstore(x0 + t1, y_ptr + idx)
    vstore(x2 - x1, y_ptr + idx + s * 16)
    vstore(x2 + x1, y_ptr + idx + 2s * 16)
end

@inline function fft3_codelet_layered!(x_ptr::Ptr{Float64}, y_ptr::Ptr{Float64}, q::Int, s::Int, p::Int, n1::Int)
    idx = (q - 1) * 16
    
    x0 = vload(Vec{2, Float64}, x_ptr + idx + s*p*16)
    x1 = vload(Vec{2, Float64}, x_ptr + idx + s * (p + n1) * 16)
    x2 = vload(Vec{2, Float64}, x_ptr + idx + s * (p + 2n1) * 16)

    t1 = x1 + x2
    x1 = x2 - x1
    x1 = Sp * Vec(x1[2], -x1[1])
    x2 = x0 - t1*Cp

    vstore(x0 + t1, y_ptr + idx + s*3p*16)
    vstore(x2 - x1, y_ptr + idx + s * (3p + 1) * 16)
    vstore(x2 + x1, y_ptr + idx + s * (3p + 2) * 16)
end

@inline function simd_3_codelet!(y_ptr::Ptr{Float64}, theta::Float64, s::Int, p::Int)
    p_theta = -p * theta

    cos_p_theta = SIMD.cospi(p_theta)
    sin_p_theta = SIMD.sinpi(p_theta)
    cos_2p_theta = SIMD.cospi(2p_theta)
    sin_2p_theta = SIMD.sinpi(2p_theta)

    @inbounds @simd for q in 1:s
        idx = (q - 1)*16

        y_val1 = vload(Vec{2, Float64}, y_ptr + idx + s*(3p + 1)*16)
        y_val2 = vload(Vec{2, Float64}, y_ptr + idx + s*(3p + 2)*16)

        real_part1 = y_val1[1] * cos_p_theta - y_val1[2] * sin_p_theta
        imag_part1 = y_val1[1] * sin_p_theta + y_val1[2] * cos_p_theta
        real_part2 = y_val2[1] * cos_2p_theta - y_val2[2] * sin_2p_theta
        imag_part2 = y_val2[1] * sin_2p_theta + y_val2[2] * cos_2p_theta

        vstore(Vec(real_part1, imag_part1), y_ptr + idx + s*(3p + 1)*16)
        vstore(Vec(real_part2, imag_part2), y_ptr + idx + s*(3p + 2)*16)
    end
end

end