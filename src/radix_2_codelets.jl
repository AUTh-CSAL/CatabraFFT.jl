module radix2_family
using LoopVectorization, SIMD

const INV_SQRT2_DEFAULT = 0.7071067811865475
const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)

# Core FFT codelets
@inline function fft16_shell_layered!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int, n1::Int, theta::Float64) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)
    @inbounds @simd for p in 0:(n1-1)
        w1p = cispi(-p * theta)
        w2p = w1p * w1p
        w3p = w1p * w2p
        w4p = w2p * w2p
        w5p = w1p * w4p
        w6p = w2p * w4p
        w7p = w3p * w4p
        w8p = w4p * w4p
        w9p = w5p * w4p
        w10p = w5p * w5p
        w11p = w5p * w6p
        w12p = w6p * w6p
        w13p = w6p * w7p
        w14p = w8p * w6p
        w15p = w7p * w8p
        @inbounds @simd for q in 1:s

            t45, t46 = x[q+s*p] + x[q+s*(p+8n1)], x[q+s*p] - x[q+s*(p+8n1)]
            t47, t48 = (x[q+s*(p+4n1)] + x[q+s*(p+12n1)]), -im * (x[q+s*(p+4n1)] - x[q+s*(p+12n1)])
            t37, t38 = t45 + t47, t46 + t48
            t39, t40 = t45 - t47, t46 - t48

            t49, t50 = x[q+s*(p+2n1)] + x[q+s*(p+10n1)], x[q+s*(p+2n1)] - x[q+s*(p+10n1)]
            t51, t52 = (x[q+s*(p+6n1)] + x[q+s*(p+14n1)]), -im * (x[q+s*(p+6n1)] - x[q+s*(p+14n1)])
            t41, t42 = (t49 + t51), INV_SQRT2 * (1 - im) * (t50 + t52)
            t43, t44 = -im * (t49 - t51), INV_SQRT2 *(-1 -im) * (t50 - t52)

            t21, t22, t23, t24 = t37 + t41, t38 + t42, t39 + t43, t40 + t44
            t25, t26, t27, t28 = t37 - t41, t38 - t42, t39 - t43, t40 - t44

            t61, t62 = x[q+s*(p+n1)] + x[q+s*(p+9n1)], x[q+s*(p+n1)] - x[q+s*(p+9n1)]
            t63, t64 = (x[q+s*(p+5n1)] + x[q+s*(p+13n1)]), -im * (x[q+s*(p+5n1)] - x[q+s*(p+13n1)])
            t53, t54 = t61 + t63, t62 + t64
            t55, t56 = t61 - t63, t62 - t64

            t65, t66 = x[q+s*(p+3n1)] + x[q+s*(p+11n1)], x[q+s*(p+3n1)] - x[q+s*(p+11n1)]
            t67, t68 = (x[q+s*(p+7n1)] + x[q+s*(p+15n1)]), -im * (x[q+s*(p+7n1)] - x[q+s*(p+15n1)])
            t57, t58 = (t65 + t67), INV_SQRT2 * (1 - im) * (t66 + t68)
            t59, t60 = -im * (t65 - t67), INV_SQRT2 * (-1 -im) * (t66 - t68)

            t29, t30, t31, t32 = (t53 + t57), (Sp_3_8 - Cp_3_8 * im) * (t54 + t58), INV_SQRT2 * (1 - im) * (t55 + t59), (Cp_3_8 - Sp_3_8 * im) * (t56 + t60)

            t33, t34, t35, t36 = -im * (t53 - t57), (-Cp_3_8 - Sp_3_8 * im) * (t54 - t58), INV_SQRT2 * (-1 -im) * (t55 - t59), (-Sp_3_8 - Cp_3_8 * im) * (t56 - t60)

            y[q+s*16p], y[q+s*(16p+1)], y[q+s*(16p+2)], y[q+s*(16p+3)] = t21 + t29, w1p * (t22 + t30), w2p * (t23 + t31), w3p * (t24 + t32)
            y[q+s*(16p+4)], y[q+s*(16p+5)], y[q+s*(16p+6)], y[q+s*(16p+7)] = w4p * (t25 + t33), w5p * (t26 + t34), w6p * (t27 + t35), w7p * (t28 + t36)
            y[q+s*(16p+8)], y[q+s*(16p+9)], y[q+s*(16p+10)], y[q+s*(16p+11)] = w8p * (t21 - t29), w9p * (t22 - t30), w10p * (t23 - t31), w11p * (t24 - t32)
            y[q+s*(16p+12)], y[q+s*(16p+13)], y[q+s*(16p+14)], y[q+s*(16p+15)] = w12p * (t25 - t33), w13p * (t26 - t34), w14p * (t27 - t35), w15p * (t28 - t36)
        end
    end
end

@inline function fft16_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)
    @inbounds @simd for q in 1:s
        t45, t46 = x[q] + x[q+8s], x[q] - x[q+8s]
        t47, t48 = (x[q+4s] + x[q+12s]), -im * (x[q+4s] - x[q+12s])
        t37, t38 = t45 + t47, t46 + t48
        t39, t40 = t45 - t47, t46 - t48

        t49, t50 = x[q+2s] + x[q+10s], x[q+2s] - x[q+10s]
        t51, t52 = (x[q+6s] + x[q+14s]), -im * (x[q+6s] - x[q+14s])
        t41, t42 = (t49 + t51), INV_SQRT2 * (1 - im) * (t50 + t52)
        t43, t44 = -im * (t49 - t51), INV_SQRT2 * (-1 -im) * (t50 - t52)

        t21, t22, t23, t24 = t37 + t41, t38 + t42, t39 + t43, t40 + t44
        t25, t26, t27, t28 = t37 - t41, t38 - t42, t39 - t43, t40 - t44

        t61, t62 = x[q+s] + x[q+9s], x[q+s] - x[q+9s]
        t63, t64 = (x[q+5s] + x[q+13s]), -im * (x[q+5s] - x[q+13s])
        t53, t54 = t61 + t63, t62 + t64
        t55, t56 = t61 - t63, t62 - t64

        t65, t66 = x[q+3s] + x[q+11s], x[q+3s] - x[q+11s]
        t67, t68 = (x[q+7s] + x[q+15s]), -im * (x[q+7s] - x[q+15s])
        t57, t58 = (t65 + t67), INV_SQRT2 * (1 - im) * (t66 + t68)
        t59, t60 = -im * (t65 - t67), INV_SQRT2 * (-1 - im) * (t66 - t68)

        t29, t30, t31, t32 = (t53 + t57), (Sp_3_8 - Cp_3_8 * im) * (t54 + t58), INV_SQRT2 * (1 - im) * (t55 + t59), (Cp_3_8 - Sp_3_8 * im) * (t56 + t60)

        t33, t34, t35, t36 = -im * (t53 - t57), (-Cp_3_8 - Sp_3_8 * im) * (t54 - t58), INV_SQRT2 * (-1 - im) * (t55 - t59), (-Sp_3_8 - Cp_3_8 * im) * (t56 - t60)

        y[q], y[q+s], y[q+2s], y[q+3s] = t21 + t29, t22 + t30, t23 + t31, t24 + t32
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t25 + t33, t26 + t34, t27 + t35, t28 + t36
        y[q+8s], y[q+9s], y[q+10s], y[q+11s] = t21 - t29, t22 - t30, t23 - t31, t24 - t32
        y[q+12s], y[q+13s], y[q+14s], y[q+15s] = t25 - t33, t26 - t34, t27 - t35, t28 - t36
    end
end

@inline function fft16_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)
    @inbounds @simd for q in 1:s
        t45, t46 = y[q] + y[q+8s], y[q] - y[q+8s]
        t47, t48 = (y[q+4s] + y[q+12s]), -im * (y[q+4s] - y[q+12s])
        t37, t38 = t45 + t47, t46 + t48
        t39, t40 = t45 - t47, t46 - t48

        t49, t50 = y[q+2s] + y[q+10s], y[q+2s] - y[q+10s]
        t51, t52 = (y[q+6s] + y[q+14s]), -im * (y[q+6s] - y[q+14s])
        t41, t42 = (t49 + t51), INV_SQRT2 * (1 - im) * (t50 + t52)
        t43, t44 = -im * (t49 - t51), INV_SQRT2 * (-1 - im) * (t50 - t52)

        t21, t22, t23, t24 = t37 + t41, t38 + t42, t39 + t43, t40 + t44
        t25, t26, t27, t28 = t37 - t41, t38 - t42, t39 - t43, t40 - t44

        t61, t62 = y[q+s] + y[q+9s], y[q+s] - y[q+9s]
        t63, t64 = (y[q+5s] + y[q+13s]), -im * (y[q+5s] - y[q+13s])
        t53, t54 = t61 + t63, t62 + t64
        t55, t56 = t61 - t63, t62 - t64

        t65, t66 = y[q+3s] + y[q+11s], y[q+3s] - y[q+11s]
        t67, t68 = (y[q+7s] + y[q+15s]), -im * (y[q+7s] - y[q+15s])
        t57, t58 = (t65 + t67), INV_SQRT2 * (1 - im) * (t66 + t68)
        t59, t60 = -im * (t65 - t67), INV_SQRT2 * (-1 - im) * (t66 - t68)

        t29, t30, t31, t32 = (t53 + t57), (Sp_3_8 - Cp_3_8 * im) * (t54 + t58), INV_SQRT2 * (1 - im) * (t55 + t59), (Cp_3_8 - Sp_3_8 * im) * (t56 + t60)

        t33, t34, t35, t36 = -im * (t53 - t57), (-Cp_3_8 - Sp_3_8 * im) * (t54 - t58), INV_SQRT2 * (-1 - im) * (t55 - t59), (-Sp_3_8 - Cp_3_8 * im) * (t56 - t60)

        y[q], y[q+s], y[q+2s], y[q+3s] = t21 + t29, t22 + t30, t23 + t31, t24 + t32
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t25 + t33, t26 + t34, t27 + t35, t28 + t36
        y[q+8s], y[q+9s], y[q+10s], y[q+11s] = t21 - t29, t22 - t30, t23 - t31, t24 - t32
        y[q+12s], y[q+13s], y[q+14s], y[q+15s] = t25 - t33, t26 - t34, t27 - t35, t28 - t36
    end
end

@inline function fft8_shell_layered_theta_1_8!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int, n1::Int) where T<:AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)
    @inbounds @simd for q in 1:s
        t13, t14 = x[q] + x[q+s*4n1], x[q] - x[q+s*4n1]
        t15, t16 = (x[q+s*2n1] + x[q+s*6n1]), -im * (x[q+s*2n1] - x[q+s*6n1])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s*n1] + x[q+s*5n1], x[q+s*n1] - x[q+s*5n1]
        t19, t20 = (x[q+s*3n1] + x[q+s*7n1]), -im * (x[q+s*3n1] - x[q+s*7n1])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 - im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end
    @inbounds @simd for q in 1:s
        t13, t14 = x[q+s] + x[q+s*(1+4n1)], x[q+s] - x[q+s*(1+4n1)]
        t15, t16 = (x[q+s*(1+2n1)] + x[q+s*(1+6n1)]), -im * (x[q+s*(1+2n1)] - x[q+s*(1+6n1)])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s*(1+n1)] + x[q+s*(1+5n1)], x[q+s*(1+n1)] - x[q+s*(1+5n1)]
        t19, t20 = (x[q+s*(1+3n1)] + x[q+s*(1+7n1)]), -im * (x[q+s*(1+3n1)] - x[q+s*(1+7n1)])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 - im) * (t18 - t20)

        y[q+s*8], y[q+s*9], y[q+s*10], y[q+s*11] = t5 + t9, (t6 + t10) * (Sp_3_8 - im * Cp_3_8), INV_SQRT2 * (t7 + t11) * (1 - im), (t8 + t12) * (Cp_3_8 - im * Sp_3_8)
        y[q+s*12], y[q+s*13], y[q+s*14], y[q+s*15] = -im * (t5 - t9), (t6 - t10) * (-Cp_3_8 - im * Sp_3_8), INV_SQRT2 * (t7 - t11) * (-1 - im), (t8 - t12) * (-Sp_3_8 - im * Cp_3_8)
    end
end

@inline function fft8_shell_layered!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int, n1::Int, theta::Float64) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)

    @inbounds @simd for q in 1:s
        t13, t14 = x[q] + x[q+s*4n1], x[q] - x[q+s*4n1]
        t15, t16 = (x[q+s*2n1] + x[q+s*6n1]), -im * (x[q+s*2n1] - x[q+s*6n1])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s*n1] + x[q+s*5n1], x[q+s*n1] - x[q+s*5n1]
        t19, t20 = (x[q+s*3n1] + x[q+s*7n1]), -im * (x[q+s*3n1] - x[q+s*7n1])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 - im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end

    @inbounds @simd for p in 1:(n1-1)
        w1p = cispi(T(-p * theta))
        w2p = w1p * w1p
        w3p = w1p * w2p
        w4p = w2p * w2p
        w5p = w2p * w3p
        w6p = w3p * w3p
        w7p = w3p * w4p
        @inbounds @simd for q in 1:s
            t13, t14 = x[q+s*p] + x[q+s*(p+4n1)], x[q+s*p] - x[q+s*(p+4n1)]
            t15, t16 = (x[q+s*(p+2n1)] + x[q+s*(p+6n1)]), -im * (x[q+s*(p+2n1)] - x[q+s*(p+6n1)])

            t5, t6 = t13 + t15, t14 + t16
            t7, t8 = t13 - t15, t14 - t16

            t17, t18 = x[q+s*(p+n1)] + x[q+s*(p+5n1)], x[q+s*(p+n1)] - x[q+s*(p+5n1)]
            t19, t20 = (x[q+s*(p+3n1)] + x[q+s*(p+7n1)]), -im * (x[q+s*(p+3n1)] - x[q+s*(p+7n1)])

            t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
            t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 - im) * (t18 - t20)

            y[q+s*8p], y[q+s*(8p+1)], y[q+s*(8p+2)], y[q+s*(8p+3)] = t5 + t9, (t6 + t10) * w1p, (t7 + t11) * w2p, (t8 + t12) * w3p
            y[q+s*(8p+4)], y[q+s*(8p+5)], y[q+s*(8p+6)], y[q+s*(8p+7)] = (t5 - t9) * w4p, (t6 - t10) * w5p, (t7 - t11) * w6p, (t8 - t12) * w7p
        end
    end
end

#=
@inline function sfft8_shell!(x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    t13, t14 = x[1] + x[5], x[1] - x[5]
    t15, t16 = x[3] + x[7], -im * (x[3] - x[7])

    t5, t6 = t13 + t15, t14 + t16
    t7, t8 = t13 - t15, t14 - t16

    t17, t18 = x[2] + x[6], x[1] - x[6]
    t19, t20 = x[4] + x[8], -im * (x[4] - x[8])

    t9, t10 = (t17 + t19), INV_SQRT2 * (1 - im) * (t18 + t20)
    t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 -im) * (t18 - t20)
    return SVector(t5 + t9, t6 + t10, t7 + t11, t8 + t12, t5 - t9, t6 - t10, t7 - t11, t8 - t12)
end
=#

@inline function fft8_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    @inbounds @simd for q in 1:s
        t13, t14 = x[q] + x[q+4s], x[q] - x[q+4s]
        t15, t16 = (x[q+2s] + x[q+6s]), -im * (x[q+2s] - x[q+6s])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = x[q+s] + x[q+5s], x[q+s] - x[q+5s]
        t19, t20 = (x[q+3s] + x[q+7s]), -im * (x[q+3s] - x[q+7s])

        t9, t10 = (t17 + t19), INV_SQRT2 * (1 -im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), INV_SQRT2 * (-1 -im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end
end

@inline function fft8_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    @inbounds @simd for q in 1:s
        t13, t14 = y[q] + y[q+4s], y[q] - y[q+4s]
        t15, t16 = (y[q+2s] + y[q+6s]), -im * (y[q+2s] - y[q+6s])

        t5, t6 = t13 + t15, t14 + t16
        t7, t8 = t13 - t15, t14 - t16

        t17, t18 = y[q+s] + y[q+5s], y[q+s] - y[q+5s]
        t19, t20 = (y[q+3s] + y[q+7s]), -im * (y[q+3s] - y[q+7s])

        t9, t10 = (t17 + t19), (INV_SQRT2 - INV_SQRT2 * im) * (t18 + t20)
        t11, t12 = -im * (t17 - t19), (-INV_SQRT2 - INV_SQRT2 * im) * (t18 - t20)

        y[q], y[q+s], y[q+2s], y[q+3s] = t5 + t9, t6 + t10, t7 + t11, t8 + t12
        y[q+4s], y[q+5s], y[q+6s], y[q+7s] = t5 - t9, t6 - t10, t7 - t11, t8 - t12
    end
end

#=
@inline function sfft4_shell!(x::AbstractVector{Complex{T}}) where T <: AbstractFloat
    t1, t2 = x[1] + x[3], x[1] - x[3]
    t3, t4 = x[2] + x[4], -im * (x[1] - x[4])
    return SVector(t1 + t3, t2 + t4, t1 - t3, t2 - t4)
end
=#

@inline function fft4_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        t1, t2 = x[q] + x[q+2s], x[q] - x[q+2s]
        t3, t4 = (x[q+s] + x[q+3s]), -im * (x[q+s] - x[q+3s])
        y[q], y[q+s] = t1 + t3, t2 + t4
        y[q+2s], y[q+3s] = t1 - t3, t2 - t4
    end
end

@inline function fft4_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        t1, t2 = y[q] + y[q+2s], y[q] - y[q+2s]
        t3, t4 = (y[q+s] + y[q+3s]), -im * (y[q+s] - y[q+3s])
        y[q], y[q+s] = t1 + t3, t2 + t4
        y[q+2s], y[q+3s] = t1 - t3, t2 - t4
    end
end

# out-of-place
#=
@inline function sfft2_shell!(y::SVector{2, Complex{T}}, x::SVector{2, Complex{T}}) where T <: AbstractFloat
    a, b = x[1], x[2]
    y = SVector(a + b, a - b)
end
=#

@inline function fft2_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        a, b = x[q], x[q+s]
        y[q] = a + b
        y[q+s] = a - b
    end
end

@inline function fft2_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    @inbounds @simd for q in 1:s
        a, b = y[q], y[q+s]
        y[q] = a + b
        y[q+s] = a - b
    end
end
end

module radix2_simd_family

using LoopVectorization, SIMD
const INV_SQRT2_DEFAULT = 0.7071067811865475
const Cp_3_8_DEFAULT = 0.3826834323650898 # cospi(3/8)
const Sp_3_8_DEFAULT = 0.9238795325112867 # sinpi(3/8)

# HW explicit for AVX2 - 256 bits / 2 ComplexF64 => 4 Float64
@inline function fft2_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})

    @inbounds @simd for q in 0:s-1
        x_idx1, x_idx2 = ptr + q * FLOAT_SIZE, ptr + (q + s) * FLOAT_SIZE
        # Load 2 complex numbers (256 bits) from each half
        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)

        sum = a + b
        diff = a - b

        vstore(sum, x_idx1)
        vstore(diff, x_idx2)
    end
    return nothing
end

@inline function fft2_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = pointer(reinterpret(T, x))
    y_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})

    @inbounds @simd for q in 0:s-1
        x_idx1, x_idx2 = x_ptr + q * FLOAT_SIZE, x_ptr + (q + s) * FLOAT_SIZE
        y_idx1, y_idx2 = y_ptr + q * FLOAT_SIZE, y_ptr + (q + s) * FLOAT_SIZE
        # Load 2 complex numbers (256 bits) from each half
        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)

        sum = a + b
        diff = a - b

        vstore(sum, y_idx1)
        vstore(diff, y_idx2)
    end
    return nothing
end

@inline function fft4_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})

    @inbounds @simd for q in 0:s-1
        # Pointers to real/imag parts of complex numbers (stored as Float64)
        x_idx1 = ptr + q * FLOAT_SIZE
        x_idx2 = ptr + (q + s) * FLOAT_SIZE
        x_idx3 = ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = ptr + (q + 3s) * FLOAT_SIZE

        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)

        t1 = a + c
        t2 = a - c
        t3 = b + d
        t4 = Vec(b[2] - d[2] , -b[1] + d[1])

        # Store the results back
        vstore(t1 + t3, x_idx1)
        vstore(t2 + t4, x_idx2)
        vstore(t1 - t3, x_idx3)
        vstore(t2 - t4, x_idx4)
    end
    return nothing
end

@inline function fft4_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = pointer(reinterpret(T, x))
    y_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})

    @inbounds @simd for q in 0:s-1
        x_idx1 = x_ptr + q * FLOAT_SIZE
        x_idx2 = x_ptr + (q + s) * FLOAT_SIZE
        x_idx3 = x_ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = x_ptr + (q + 3s) * FLOAT_SIZE

        y_idx1 = y_ptr + q * FLOAT_SIZE
        y_idx2 = y_ptr + (q + s) * FLOAT_SIZE
        y_idx3 = y_ptr + (q + 2s) * FLOAT_SIZE
        y_idx4 = y_ptr + (q + 3s) * FLOAT_SIZE

        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)

        t1 = a + c
        t2 = a - c
        t3 = b + d
        t4 = Vec(b[2] - d[2] , -b[1] + d[1])

        vstore(t1 + t3, y_idx1)
        vstore(t2 + t4, y_idx2)
        vstore(t1 - t3, y_idx3)
        vstore(t2 - t4, y_idx4)
    end
    return nothing
end

@inline function fft8_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)

    @inbounds @simd for q in 0:s-1
        x_idx1 = ptr + (q) * FLOAT_SIZE
        x_idx2 = ptr + (q + s) * FLOAT_SIZE
        x_idx3 = ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = ptr + (q + 3s) * FLOAT_SIZE
        x_idx5 = ptr + (q + 4s) * FLOAT_SIZE
        x_idx6 = ptr + (q + 5s) * FLOAT_SIZE
        x_idx7 = ptr + (q + 6s) * FLOAT_SIZE
        x_idx8 = ptr + (q + 7s) * FLOAT_SIZE

        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)
        e = vload(Vec{2, T}, x_idx5)
        f = vload(Vec{2, T}, x_idx6)
        g = vload(Vec{2, T}, x_idx7)
        h = vload(Vec{2, T}, x_idx8)

        t13 = a + e
        t14 = a - e
        t15 = c + g
        t17 = b + f
        t18 = b - f
        t19 = d + h
        t16 = Vec(c[2] - g[2], -c[1] + g[1])

        t5 = t13 + t15
        t6 = t14 + t16
        t7 = t13 - t15
        t8 = t14 - t16
        t9 = t17 + t19

        t20 = Vec(d[2] - h[2], -d[1] + h[1])
        t10_real = INV_SQRT2 * (t18[1] + t20[1] - t18[2] - t20[2])
        t10 = Vec(t10_real, -t10_real)
        t11 = Vec(t17[2] - t19[2], -t17[1] + t19[1])
        t12_real = t18[2] - t20[2] - t18[1] + t20[1]
        t12_imag = -t18[1] + t20[1] - t18[1] + t20[2]
        t12 = INV_SQRT2 * Vec(t12_real, t12_imag)

        vstore(t5 + t9, x_idx1)
        vstore(t6 + t10, x_idx2)
        vstore(t7 + t11, x_idx3)
        vstore(t8 + t12, x_idx4)

        vstore(t5 - t9, x_idx5)
        vstore(t6 - t10, x_idx6)
        vstore(t7 - t11, x_idx7)
        vstore(t8 - t12, x_idx8)
    end
    return nothing
end


@inline function fft8_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = pointer(reinterpret(T, x))  # Get pointer to the start of the array
    y_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)

    @inbounds @simd for q in 0:s-1
        # Load data for 8 complex numbers from x
        x_idx1 = x_ptr + (q) * FLOAT_SIZE
        x_idx2 = x_ptr + (q + s) * FLOAT_SIZE
        x_idx3 = x_ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = x_ptr + (q + 3s) * FLOAT_SIZE
        x_idx5 = x_ptr + (q + 4s) * FLOAT_SIZE
        x_idx6 = x_ptr + (q + 5s) * FLOAT_SIZE
        x_idx7 = x_ptr + (q + 6s) * FLOAT_SIZE
        x_idx8 = x_ptr + (q + 7s) * FLOAT_SIZE

        y_idx1 = y_ptr + (q) * FLOAT_SIZE
        y_idx2 = y_ptr + (q + s) * FLOAT_SIZE
        y_idx3 = y_ptr + (q + 2s) * FLOAT_SIZE
        y_idx4 = y_ptr + (q + 3s) * FLOAT_SIZE
        y_idx5 = y_ptr + (q + 4s) * FLOAT_SIZE
        y_idx6 = y_ptr + (q + 5s) * FLOAT_SIZE
        y_idx7 = y_ptr + (q + 6s) * FLOAT_SIZE
        y_idx8 = y_ptr + (q + 7s) * FLOAT_SIZE

        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)
        e = vload(Vec{2, T}, x_idx5)
        f = vload(Vec{2, T}, x_idx6)
        g = vload(Vec{2, T}, x_idx7)
        h = vload(Vec{2, T}, x_idx8)

        t13 = a + e
        t14 = a - e
        t15 = c + g
        t17 = b + f
        t18 = b - f
        t19 = d + h
        t16 = Vec(c[2] - g[2], -c[1] + g[1])

        t5 = t13 + t15
        t6 = t14 + t16
        t7 = t13 - t15
        t8 = t14 - t16
        t9 = t17 + t19

        t20 = Vec(d[2] - h[2], -d[1] + h[1])
        t10_real = INV_SQRT2 * (t18[1] + t20[1] - t18[2] - t20[2])
        t10 = Vec(t10_real, -t10_real)
        t11 = Vec(t17[2] - t19[2], -t17[1] + t19[1])
        t12_real = t18[2] - t20[2] - t18[1] + t20[1]
        t12_imag = -t18[1] + t20[1] - t18[1] + t20[2]
        t12 = INV_SQRT2 * Vec(t12_real, t12_imag)

        vstore(t5 + t9, y_idx1)
        vstore(t6 + t10, y_idx2)
        vstore(t7 + t11, y_idx3)
        vstore(t8 + t12, y_idx4)

        vstore(t5 - t9, y_idx5)
        vstore(t6 - t10, y_idx6)
        vstore(t7 - t11, y_idx7)
        vstore(t8 - t12, y_idx8)
    end
    return nothing
end

@inline function fft16_shell_y!(y::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat
    x_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)

    @inbounds @simd for q in 0:s-1
        x_idx1 = x_ptr + (q) * FLOAT_SIZE
        x_idx2 = x_ptr + (q + s) * FLOAT_SIZE
        x_idx3 = x_ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = x_ptr + (q + 3s) * FLOAT_SIZE
        x_idx5 = x_ptr + (q + 4s) * FLOAT_SIZE
        x_idx6 = x_ptr + (q + 5s) * FLOAT_SIZE
        x_idx7 = x_ptr + (q + 6s) * FLOAT_SIZE
        x_idx8 = x_ptr + (q + 7s) * FLOAT_SIZE
        x_idx9 = x_ptr + (q + 8s) * FLOAT_SIZE
        x_idx10 = x_ptr + (q + 9s) * FLOAT_SIZE
        x_idx11 = x_ptr + (q + 10s) * FLOAT_SIZE
        x_idx12 = x_ptr + (q + 11s) * FLOAT_SIZE
        x_idx13 = x_ptr + (q + 12s) * FLOAT_SIZE
        x_idx14 = x_ptr + (q + 13s) * FLOAT_SIZE
        x_idx15 = x_ptr + (q + 14s) * FLOAT_SIZE
        x_idx16 = x_ptr + (q + 15s) * FLOAT_SIZE

        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)
        e = vload(Vec{2, T}, x_idx5)
        f = vload(Vec{2, T}, x_idx6)
        g = vload(Vec{2, T}, x_idx7)
        h = vload(Vec{2, T}, x_idx8)
        i = vload(Vec{2, T}, x_idx9)
        j = vload(Vec{2, T}, x_idx10)
        k = vload(Vec{2, T}, x_idx11)
        l = vload(Vec{2, T}, x_idx12)
        m = vload(Vec{2, T}, x_idx13)
        n = vload(Vec{2, T}, x_idx14)
        o = vload(Vec{2, T}, x_idx15)
        p = vload(Vec{2, T}, x_idx16)

        t45, t46 = a + i, a - i
        t47 = e + m
        t48 = Vec(e[2] - m[2], -e[1] + m[1])
        t37, t38 = t45 + t47, t46 + t48
        t39, t40 = t45 - t47, t46 - t48

        t49, t50 = c + k, c - k
        t51 = g + o
        t52 = Vec(g[2] - o[2], -g[1] + m[1])
        t41 = t49 + t51
        t42_real = INV_SQRT2 * (t50[1] + t52[1] - t50[2] - t52[2])
        t42 = Vec(t42_real, -t42_real)
        t43 = Vec(t49[2] - t51[2], -t49[1] + t51[1])
        t44_real = INV_SQRT2 * (t50[2] - t52[2] - t50[1] + t52[1])
        t44_imag = INV_SQRT2 * (-t50[1] + t52[1] - t50[2] + t52[2])
        t44 = Vec(t44_real, t44_imag)

        t21, t22, t23, t24 = t37 + t41, t38 + t42, t39 + t43, t40 + t44
        t25, t26, t27, t28 = t37 - t41, t38 - t42, t39 - t43, t40 - t44

        t61, t62 = b + j, b - j
        t63, t64 = f + n, Vec(f[2] - n[2], -f[1] + n[1])
        t53, t54 = t61 + t63, t62 + t64
        t55, t56 = t61 - t63, t62 - t64

        t65, t66 = d + l, d - l
        t67, t68 = h + p, Vec(h[2] - p[2], -h[1] + p[1])
        t57 = t65 + t67
        t58_real = INV_SQRT2 * (t66[1] + t68[1] - t66[2] - t68[2])
        t58 = Vec(t58_real, -t58_real)
        t59 = Vec(t65[2] - t67[2], -t65[1] + t67[1])

        t60_real = INV_SQRT2 * (t66[2] - t68[2] - t66[1] + t68[1])
        t60_imag = INV_SQRT2 * (-t66[1] + t68[1] - t66[2] + t68[2])
        t60 = Vec(t60_real, t60_imag)

        t29 = t53 + t57
        t_sum = t54 + t58
        t30_real = Sp_3_8 * t_sum[1] + Cp_3_8 * t_sum[2]
        t30_imag = Sp_3_8 * t_sum[2] + Cp_3_8 * t_sum[1] #t_diff[1]
        t30 = Vec(t30_real, t30_imag)
        t_sum = t55 + t59
        t31_real = INV_SQRT2 * (t_sum[1] - t_sum[2])
        t31_imag = INV_SQRT2 * (t_sum[1] + t_sum[2])
        t31 = Vec(t31_real, t31_imag)
        t_sum = t56 + t60
        t32_real = Cp_3_8 * t_sum[1] + Sp_3_8 * t_sum[2]
        t32_imag = Cp_3_8 * t_sum[2] - Sp_3_8 * t_sum[1]
        t32 = Vec(t32_real, t32_imag)

        t33 = Vec(t53[2] - t57[2], -t53[1] + t57[1])
        t_diff = t54 - t58
        t34_real = -Cp_3_8 * t_diff[1] - Sp_3_8 * t_diff[2]
        t34_imag = -Cp_3_8 * t_diff[2] + Sp_3_8 * t_diff[1]
        t34 = Vec(t34_real, t34_imag)
        t_diff = t55 - t59
        t35_real = -INV_SQRT2 * (t_diff[1] + t_diff[2])
        t35_imag = -INV_SQRT2 * (t_diff[1] - t_diff[2])
        t35 = Vec(t35_real, t35_imag)
        t_diff = t56 - t60
        t36_real = + Cp_3_8 * t_diff[1] - Sp_3_8 * t_diff[2]
        t36_imag = - Cp_3_8 * t_diff[2] + Sp_3_8 * t_diff[1]
        t36 = Vec(t36_real, t36_imag)

        vstore(t21 + t29, x_idx1)
        vstore(t22 + t30, x_idx2)
        vstore(t23 + t31, x_idx3)
        vstore(t24 + t32, x_idx4)
        vstore(t25 + t33, x_idx5)
        vstore(t26 + t34, x_idx6)
        vstore(t27 + t35, x_idx7)
        vstore(t28 + t36, x_idx8)

        vstore(t21 - t29, x_idx9)
        vstore(t22 - t30, x_idx10)
        vstore(t23 - t31, x_idx11)
        vstore(t24 - t32, x_idx12)
        vstore(t25 - t33, x_idx13)
        vstore(t26 - t34, x_idx14)
        vstore(t27 - t35, x_idx15)
        vstore(t28 - t36, x_idx16)
    end
    return nothing
end

@inline function fft16_shell!(y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}, s::Int) where T <: AbstractFloat 
    x_ptr = pointer(reinterpret(T, x))
    y_ptr = pointer(reinterpret(T, y))
    FLOAT_SIZE = sizeof(Complex{T})
    INV_SQRT2 = T(INV_SQRT2_DEFAULT)
    Cp_3_8 = T(Cp_3_8_DEFAULT)
    Sp_3_8 = T(Sp_3_8_DEFAULT)

    @inbounds @simd for q in 0:s-1
        x_idx1 = x_ptr + (q) * FLOAT_SIZE
        x_idx2 = x_ptr + (q + s) * FLOAT_SIZE
        x_idx3 = x_ptr + (q + 2s) * FLOAT_SIZE
        x_idx4 = x_ptr + (q + 3s) * FLOAT_SIZE
        x_idx5 = x_ptr + (q + 4s) * FLOAT_SIZE
        x_idx6 = x_ptr + (q + 5s) * FLOAT_SIZE
        x_idx7 = x_ptr + (q + 6s) * FLOAT_SIZE
        x_idx8 = x_ptr + (q + 7s) * FLOAT_SIZE
        x_idx9 = x_ptr + (q + 8s) * FLOAT_SIZE
        x_idx10 = x_ptr + (q + 9s) * FLOAT_SIZE
        x_idx11 = x_ptr + (q + 10s) * FLOAT_SIZE
        x_idx12 = x_ptr + (q + 11s) * FLOAT_SIZE
        x_idx13 = x_ptr + (q + 12s) * FLOAT_SIZE
        x_idx14 = x_ptr + (q + 13s) * FLOAT_SIZE
        x_idx15 = x_ptr + (q + 14s) * FLOAT_SIZE
        x_idx16 = x_ptr + (q + 15s) * FLOAT_SIZE

        # Load complex numbers using SIMD
        a = vload(Vec{2, T}, x_idx1)
        b = vload(Vec{2, T}, x_idx2)
        c = vload(Vec{2, T}, x_idx3)
        d = vload(Vec{2, T}, x_idx4)
        e = vload(Vec{2, T}, x_idx5)
        f = vload(Vec{2, T}, x_idx6)
        g = vload(Vec{2, T}, x_idx7)
        h = vload(Vec{2, T}, x_idx8)
        i = vload(Vec{2, T}, x_idx9)
        j = vload(Vec{2, T}, x_idx10)
        k = vload(Vec{2, T}, x_idx11)
        l = vload(Vec{2, T}, x_idx12)
        m = vload(Vec{2, T}, x_idx13)
        n = vload(Vec{2, T}, x_idx14)
        o = vload(Vec{2, T}, x_idx15)
        p = vload(Vec{2, T}, x_idx16)

        t45, t46 = a + i, a - i
        t47 = e + m
        t48 = Vec(e[2] - m[2], -e[1] + m[1])
        t37, t38 = t45 + t47, t46 + t48
        t39, t40 = t45 - t47, t46 - t48

        t49, t50 = c + k, c - k
        t51 = g + o
        t52 = Vec(g[2] - o[2], -g[1] + m[1])
        t41 = t49 + t51
        t42_real = INV_SQRT2 * (t50[1] + t52[1] - t50[2] - t52[2])
        t42 = Vec(t42_real, -t42_real)
        t43 = Vec(t49[2] - t51[2], -t49[1] + t51[1])
        t44_real = INV_SQRT2 * (t50[2] - t52[2] - t50[1] + t52[1])
        t44_imag = INV_SQRT2 * (-t50[1] + t52[1] - t50[2] + t52[2])
        t44 = Vec(t44_real, t44_imag)

        t21, t22, t23, t24 = t37 + t41, t38 + t42, t39 + t43, t40 + t44
        t25, t26, t27, t28 = t37 - t41, t38 - t42, t39 - t43, t40 - t44

        t61, t62 = b + j, b - j
        t63, t64 = f + n, Vec(f[2] - n[2], -f[1] + n[1])
        t53, t54 = t61 + t63, t62 + t64
        t55, t56 = t61 - t63, t62 - t64

        t65, t66 = d + l, d - l
        t67, t68 = h + p, Vec(h[2] - p[2], -h[1] + p[1])
        t57 = t65 + t67
        t58_real = INV_SQRT2 * (t66[1] + t68[1] - t66[2] - t68[2])
        t58 = Vec(t58_real, -t58_real)
        t59 = Vec(t65[2] - t67[2], -t65[1] + t67[1])

        t60_real = INV_SQRT2 * (t66[2] - t68[2] - t66[1] + t68[1])
        t60_imag = INV_SQRT2 * (-t66[1] + t68[1] - t66[2] + t68[2])
        t60 = Vec(t60_real, t60_imag)

        t29 = t53 + t57
        t_sum = t54 + t58
        t30_real = Sp_3_8 * t_sum[1] + Cp_3_8 * t_sum[2]
        t30_imag = Sp_3_8 * t_sum[2] + Cp_3_8 * t_sum[1] #t_diff[1]
        t30 = Vec(t30_real, t30_imag)
        t_sum = t55 + t59
        t31_real = INV_SQRT2 * (t_sum[1] - t_sum[2])
        t31_imag = INV_SQRT2 * (t_sum[1] + t_sum[2])
        t31 = Vec(t31_real, t31_imag)
        t_sum = t56 + t60
        t32_real = Cp_3_8 * t_sum[1] + Sp_3_8 * t_sum[2]
        t32_imag = Cp_3_8 * t_sum[2] - Sp_3_8 * t_sum[1]
        t32 = Vec(t32_real, t32_imag)

        t33 = Vec(t53[2] - t57[2], -t53[1] + t57[1])
        t_diff = t54 - t58
        t34_real = -Cp_3_8 * t_diff[1] - Sp_3_8 * t_diff[2]
        t34_imag = -Cp_3_8 * t_diff[2] + Sp_3_8 * t_diff[1]
        t34 = Vec(t34_real, t34_imag)
        t_diff = t55 - t59
        t35_real = -INV_SQRT2 * (t_diff[1] + t_diff[2])
        t35_imag = -INV_SQRT2 * (t_diff[1] - t_diff[2])
        t35 = Vec(t35_real, t35_imag)
        t_diff = t56 - t60
        t36_real = + Cp_3_8 * t_diff[1] - Sp_3_8 * t_diff[2]
        t36_imag = - Cp_3_8 * t_diff[2] + Sp_3_8 * t_diff[1]
        t36 = Vec(t36_real, t36_imag)

        vstore(t21 + t29, y_idx1)
        vstore(t22 + t30, y_idx2)
        vstore(t23 + t31, y_idx3)
        vstore(t24 + t32, y_idx4)
        vstore(t25 + t33, y_idx5)
        vstore(t26 + t34, y_idx6)
        vstore(t27 + t35, y_idx7)
        vstore(t28 + t36, y_idx8)

        vstore(t21 - t29, y_idx9)
        vstore(t22 - t30, y_idx10)
        vstore(t23 - t31, y_idx11)
        vstore(t24 - t32, y_idx12)
        vstore(t25 - t33, y_idx13)
        vstore(t26 - t34, y_idx14)
        vstore(t27 - t35, y_idx15)
        vstore(t28 - t36, y_idx16)
    end
    return nothing
end

end

