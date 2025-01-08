using Primes

include("spells.jl")

function generate_prime_fft_raders(n::Int, ::Type{T}, flag::FLAG)::Function where {T<:AbstractFloat}
    @assert isprime(n) "Input length must be prime for Rader's FFT"

    # Find primitive root
    function generator(p)
        for gen in 2:(p-1)
            powers = Set{Int}()
            current = 1
            valid = true
            for i in 1:(p-1)
                current = (current * gen) % p
                if current in powers
                    valid = false
                    break
                end
                push!(powers, current)
            end
            if valid
                perm = [powermod(gen, i, p) for i in 0:(p-2)]
                return gen, perm
            end
        end
        error("No generator found")
    end

    # Generate the generator and permutation sequence
    gen, gen_seq = generator(n)
    inv_gen = powermod(gen, n-2, n)
    inv_seq = [powermod(inv_gen, i, n) for i in 0:(n-2)]

    # Precompute twiddle factors
    ω = cispi(T(2/n))
    W = [ω^(-inv_seq[i]) for i in 1:(n-1)]

    # Get FFT for length n-1
    F = recursive_F(n-1, T, flag)

    return function (y::AbstractVector{Complex{T}}, x::AbstractVector{Complex{T}}) where {T<:AbstractFloat}
        @assert length(x) == n && length(y) == n "Input and output vectors must have length n"

        # Preallocate two buffers
        buffer1 = similar(x, n-1)
        buffer2 = similar(x, n-1)

        fill!(buffer1, zero(ComplexF64))
        fill!(buffer2, zero(ComplexF64))

        x0 = x[1]

        @inbounds @simd for i in 1:(n-1)
            buffer1[i] = x[gen_seq[i] + 1]
        end

        y[1] = x0 + sum(buffer1)

        F(buffer2, buffer1)  # FFT of permuted input -> buffer2
        F(buffer1, W)        # FFT of twiddle factors -> buffer1

        # Multiply in frequency domain (reuse buffer1 or buffer2 as needed)
        @inbounds @simd for i in 1:(n-1)
            buffer2[i] *= buffer1[i]
        end

        # Inverse FFT (reuse buffer1 for the result)
        F(buffer1, conj.(buffer2))

        @inbounds @simd for i in 1:(n-1)
            buffer1[i] = conj(buffer1[i]) / (n-1)
        end

        # Step 5: Place remaining terms
        @inbounds @simd for j in 1:(n-1)
            y[inv_seq[j] + 1] = x0 + buffer1[j]
        end

        return nothing
    end
end