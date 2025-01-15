module MatrixOperations

using LoopVectorization

# Vectorized in-place transpose for small matrices
function transpose_small!(a::AbstractArray{Complex{T}, 2}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:i-1
            tmp = a[i,j]
            a[i,j] = a[j,i]
            a[j,i] = tmp
        end
    end
end

# Cache-oblivious transpose with vectorized base case
function transpose!(a::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 8
        transpose_small!(a, n)
    else
        k = div(n, 2)
        transpose!(view(a, 1:k, 1:k), k, N)
        transpose!(view(a, 1:k, k+1:n), k, N)
        transpose!(view(a, k+1:n, 1:k), k, N)
        transpose!(view(a, k+1:n, k+1:n), k, N)
        
        @inbounds @simd for i in 1:k
            @inbounds @simd for j in 1:k
                tmp = a[i, j+k]
                a[i, j+k] = a[i+k, j]
                a[i+k, j] = tmp
            end
        end
        
        if isodd(n)
            @inbounds @simd for i in 1:n-1
                tmp = a[i,n]
                a[i,n] = a[n,i]
                a[n,i] = tmp
            end
        end
    end
end

# Vectorized base case for standard matrix multiplication
@inline function matmul_small!(a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, c::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            cij = zero(T)
            @inbounds @simd for k in 1:n
                cij += a[i,k] * b[k,j]
            end
            c[i,j] += cij
        end
    end
end

# Cache-oblivious standard matrix multiplication with vectorized base case
function matmul!(a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, c::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 16
        matmul_small!(a, b, c, n)
    else
        k = div(n, 2)
        
        matmul!(view(a, 1:k, 1:k), view(b, 1:k, 1:k), view(c, 1:k, 1:k), k, N)
        matmul!(view(a, 1:k, k+1:n), view(b, k+1:n, 1:k), view(c, 1:k, 1:k), k, N)
        
        matmul!(view(a, 1:k, 1:k), view(b, 1:k, k+1:n), view(c, 1:k, k+1:n), k, N)
        matmul!(view(a, 1:k, k+1:n), view(b, k+1:n, k+1:n), view(c, 1:k, k+1:n), k, N)
        
        matmul!(view(a, k+1:n, 1:k), view(b, 1:k, 1:k), view(c, k+1:n, 1:k), k, N)
        matmul!(view(a, k+1:n, k+1:n), view(b, k+1:n, 1:k), view(c, k+1:n, 1:k), k, N)
        
        matmul!(view(a, k+1:n, 1:k), view(b, 1:k, k+1:n), view(c, k+1:n, k+1:n), k, N)
        matmul!(view(a, k+1:n, k+1:n), view(b, k+1:n, k+1:n), view(c, k+1:n, k+1:n), k, N)
        
        if isodd(n)
            @inbounds @simd for i in 1:n
                @inbounds @simd for j in 1:n
                    @inbounds @simd for k in ((i <= n-1 && j <= n-1) ? n : 1):n
                        c[i,j] += a[i,k] * b[k,j]
                    end
                end
            end
        end
    end
end

# Strassen algorithm implementation
@inline function strassen_add!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] + b[i,j]
        end
    end
end

@inline function strassen_sub!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int) where T <: AbstractFloat
    @inbounds @simd for i in 1:n
        @inbounds @simd for j in 1:n
            c[i,j] = a[i,j] - b[i,j]
        end
    end
end

function strassen_mul!(c::AbstractMatrix{Complex{T}}, a::AbstractMatrix{Complex{T}}, b::AbstractMatrix{Complex{T}}, n::Int, N::Int) where T <: AbstractFloat
    if n <= 16  # Use standard multiplication for small matrices
        matmul_small!(a, b, c, n)
    else
        k = div(n, 2)
        
        # Temporary matrices for Strassen algorithm
        m1 = zeros(T, k, k)
        m2 = zeros(T, k, k)
        m3 = zeros(T, k, k)
        m4 = zeros(T, k, k)
        m5 = zeros(T, k, k)
        m6 = zeros(T, k, k)
        m7 = zeros(T, k, k)
        
        temp1 = zeros(T, k, k)
        temp2 = zeros(T, k, k)
        
        # M1 = (A11 + A22)(B11 + B22)
        strassen_add!(temp1, view(a, 1:k, 1:k), view(a, k+1:n, k+1:n), k)
        strassen_add!(temp2, view(b, 1:k, 1:k), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m1, temp1, temp2, k, N)
        
        # M2 = (A21 + A22)B11
        strassen_add!(temp1, view(a, k+1:n, 1:k), view(a, k+1:n, k+1:n), k)
        strassen_mul!(m2, temp1, view(b, 1:k, 1:k), k, N)
        
        # M3 = A11(B12 - B22)
        strassen_sub!(temp1, view(b, 1:k, k+1:n), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m3, view(a, 1:k, 1:k), temp1, k, N)
        
        # M4 = A22(B21 - B11)
        strassen_sub!(temp1, view(b, k+1:n, 1:k), view(b, 1:k, 1:k), k)
        strassen_mul!(m4, view(a, k+1:n, k+1:n), temp1, k, N)
        
        # M5 = (A11 + A12)B22
        strassen_add!(temp1, view(a, 1:k, 1:k), view(a, 1:k, k+1:n), k)
        strassen_mul!(m5, temp1, view(b, k+1:n, k+1:n), k, N)
        
        # M6 = (A21 - A11)(B11 + B12)
        strassen_sub!(temp1, view(a, k+1:n, 1:k), view(a, 1:k, 1:k), k)
        strassen_add!(temp2, view(b, 1:k, 1:k), view(b, 1:k, k+1:n), k)
        strassen_mul!(m6, temp1, temp2, k, N)
        
        # M7 = (A12 - A22)(B21 + B22)
        strassen_sub!(temp1, view(a, 1:k, k+1:n), view(a, k+1:n, k+1:n), k)
        strassen_add!(temp2, view(b, k+1:n, 1:k), view(b, k+1:n, k+1:n), k)
        strassen_mul!(m7, temp1, temp2, k, N)
        
        # C11 = M1 + M4 - M5 + M7
        strassen_add!(view(c, 1:k, 1:k), m1, m4, k)
        strassen_sub!(view(c, 1:k, 1:k), view(c, 1:k, 1:k), m5, k)
        strassen_add!(view(c, 1:k, 1:k), view(c, 1:k, 1:k), m7, k)
        
        # C12 = M3 + M5
        strassen_add!(view(c, 1:k, k+1:n), m3, m5, k)
        
        # C21 = M2 + M4
        strassen_add!(view(c, k+1:n, 1:k), m2, m4, k)
        
        # C22 = M1 - M2 + M3 + M6
        strassen_add!(view(c, k+1:n, k+1:n), m1, m6, k)
        strassen_sub!(view(c, k+1:n, k+1:n), view(c, k+1:n, k+1:n), m2, k)
        strassen_add!(view(c, k+1:n, k+1:n), view(c, k+1:n, k+1:n), m3, k)
    end
end
end