module radix2_family
using LoopVectorization

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
