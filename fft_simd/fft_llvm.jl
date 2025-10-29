using BenchmarkTools, FFTW, LinearAlgebra, SIMD

 @inline function vfft4(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
           @inbounds @fastmath begin
               LANE = VecRange{8}(0)
               v = x[LANE + 1]
               
               # Split into pairs
               lo = shufflevector(v, Val((0, 1, 2, 3)))  # [x1r, x1i, x2r, x2i]
               hi = shufflevector(v, Val((4, 5, 6, 7)))  # [x3r, x3i, x4r, x4i]
               
               # First butterfly
               add_vec = lo + hi  # [t1r, t1i, t3r, t3i]
               sub_vec = lo - hi  # [t2r, t2i, (x2-x4)r, (x2-x4)i]
               
               # Apply twiddle: multiply second pair by i
               # (a+bi)*i = -b+ai, so swap and negate: [(x2-x4)r, (x2-x4)i] -> [(x2-x4)i, -(x2-x4)r]
               sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2))) * Vec{4,T}((1, 1, 1, -1))
               # Now sub_vec = [t2r, t2i, t4r, t4i]
               
               # Second butterfly: need [t1,t2] ± [t3,t4]
               # Rearrange to [t1r, t1i, t2r, t2i] and [t3r, t3i, t4r, t4i]
               t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))  # [t1r, t1i, t2r, t2i]
               t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))  # [t3r, t3i, t4r, t4i]
               
               # Compute y1,y2 = t12 + t34 and y3,y4 = t12 - t34
               y12 = t12 + t34
               y34 = t12 - t34
               
               # Combine into output
               OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
               
               y[LANE + 1] = OUT
           end
       end

@inline function vfft4_opt(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = x[LANE + 1]
        
        # Split into pairs
        lo = shufflevector(v, Val((0, 1, 2, 3)))  # [x1r, x1i, x2r, x2i]
        hi = shufflevector(v, Val((4, 5, 6, 7)))  # [x3r, x3i, x4r, x4i]
        
        # First butterfly
        add_vec = lo + hi  # [t1r, t1i, t3r, t3i]
        sub_vec = lo - hi  # [t2r, t2i, (x2-x4)r, (x2-x4)i]
        
        # Apply twiddle: multiply second pair by i
        # (a+bi)*i = -b+ai, so swap and negate: [(x2-x4)r, (x2-x4)i] -> [(x2-x4)i, -(x2-x4)r]
        sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2))) * Vec{4,T}((1, 1, 1, -1))
        # Now sub_vec = [t2r, t2i, t4r, t4i]
        
        # Second butterfly: need [t1,t2] ± [t3,t4]
        # Rearrange to [t1r, t1i, t2r, t2i] and [t3r, t3i, t4r, t4i]
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))  # [t1r, t1i, t2r, t2i]
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))  # [t3r, t3i, t4r, t4i]
        
        # Compute y1,y2 = t12 + t34 and y3,y4 = t12 - t34
        y12 = t12 + t34
        y34 = t12 - t34
        
        # Combine into output
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        
        y[LANE + 1] = OUT
    end
end

@inline function signflip(v::Vec{N,T}, mask::Vec{N,UInt32}) where {N, T<:AbstractFloat}
    # Reinterpret float as uint, XOR with sign bit, reinterpret back
    v_uint = reinterpret(Vec{N,UInt32}, v)
    v_flipped = v_uint ⊻ mask
    return reinterpret(Vec{N,T}, v_flipped)
end

@inline function vfft4_xor(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = x[LANE + 1]
        
        lo = shufflevector(v, Val((0, 1, 2, 3)))
        hi = shufflevector(v, Val((4, 5, 6, 7)))
        
        add_vec = lo + hi
        sub_vec = lo - hi
        
        # Apply twiddle with XOR sign flip (flip 4th element only)
        sign_mask = Vec{4,UInt32}((0x00000000, 0x00000000, 0x00000000, 0x80000000))
        sub_vec_shuffled = shufflevector(sub_vec, Val((0, 1, 3, 2)))
        sub_vec = signflip(sub_vec_shuffled, sign_mask)
        
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))
        
        y12 = t12 + t34
        y34 = t12 - t34
        
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        y[LANE + 1] = OUT
    end
end


@inline function vfft4_best(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
           @inbounds @fastmath begin
               LANE = VecRange{8}(0)
               v = x[LANE + 1]
               
               lo = shufflevector(v, Val((0, 1, 2, 3)))
               hi = shufflevector(v, Val((4, 5, 6, 7)))
               
               add_vec = lo + hi
               sub_vec = lo - hi
               
               # Keep the winning MUL approach but try to streamline
               sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2))) * Vec{4,T}((1, 1, 1, -1))
               
               # Try different shuffle pattern
               t1 = shufflevector(add_vec, Val((0, 1)))
               t2 = shufflevector(sub_vec, Val((0, 1)))
               t3 = shufflevector(add_vec, Val((2, 3)))
               t4 = shufflevector(sub_vec, Val((2, 3)))
               
               y1 = t1 + t3
               y2 = t2 + t4
               y3 = t1 - t3
               y4 = t2 - t4
               
               OUT = shufflevector(y1, y2, y3, y4, Val((0, 1, 2, 3, 4, 5, 6, 7)))
               
               y[LANE + 1] = OUT
           end
       end

@inline function negate_element(v::Vec{4,Float32}, idx::Int)
    Base.llvmcall("""
        %elem = extractelement <4 x float> %0, i32 $(idx-1)
        %neg = fneg float %elem
        %result = insertelement <4 x float> %0, float %neg, i32 $(idx-1)
        ret <4 x float> %result
        """, Vec{4,Float32}, Tuple{Vec{4,Float32}}, v)
end

@inline function vfft4_fneg(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = x[LANE + 1]
        
        lo = shufflevector(v, Val((0, 1, 2, 3)))
        hi = shufflevector(v, Val((4, 5, 6, 7)))
        
        add_vec = lo + hi
        sub_vec = lo - hi
        
        # Apply twiddle: shuffle then negate 4th element
        sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2)))
        sub_vec = negate_element(sub_vec, 4)
        
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))
        
        y12 = t12 + t34
        y34 = t12 - t34
        
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        y[LANE + 1] = OUT
    end
end

@inline function vfft4_sub(x::Vector{T}, y::Vector{T}) where T <: AbstractFloat
    @inbounds @fastmath begin
        LANE = VecRange{8}(0)
        v = x[LANE + 1]
        
        lo = shufflevector(v, Val((0, 1, 2, 3)))
        hi = shufflevector(v, Val((4, 5, 6, 7)))
        
        add_vec = lo + hi
        sub_vec = lo - hi
        
        # Apply twiddle: shuffle, then negate 4th element via subtraction
        sub_vec = shufflevector(sub_vec, Val((0, 1, 3, 2)))
        # Extract 4th element, negate it, and put it back
        elem4 = shufflevector(sub_vec, Val((3,)))  # Extract as Vec{1}
        neg_elem4 = Vec{1,T}((zero(T),)) - elem4
        # Rebuild: [elem1, elem2, elem3, -elem4]
        sub_vec = shufflevector(sub_vec, neg_elem4, Val((0, 1, 2, 4)))
        
        t12 = shufflevector(add_vec, sub_vec, Val((0, 1, 4, 5)))
        t34 = shufflevector(add_vec, sub_vec, Val((2, 3, 6, 7)))
        
        y12 = t12 + t34
        y34 = t12 - t34
        
        OUT = shufflevector(y12, y34, Val((0, 1, 2, 3, 4, 5, 6, 7)))
        y[LANE + 1] = OUT
    end
end

x_data = ComplexF32[1+1im, 2+2im, 3+3im, 4+4im]
x = x_data
px = Vector{Float32}(reinterpret(Float32, x_data))
py = Vector{Float32}(similar(px))

# Reference
F = FFTW.plan_fft(x_data; flags=FFTW.EXHAUSTIVE)
expected = F * x

# Test
vfft4(px, py)
vfft4_xor(px, py)
result = reinterpret(ComplexF32, py)

println("Expected: ", expected)
println("Got:      ", result)
println("Match:    ", isapprox(result, expected))

# Benchmark
#@btime vfft4($px, $py)
#@benchmark vfft4($px, $py)
@benchmark vfft4_xor($px, $py)