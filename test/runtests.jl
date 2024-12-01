using CatabraFFT
using Test

@testset "Radix-2" begin
    	n = 20
	for i in 1:n
    		x = [ComplexF64(rand(), rand()) for _ in 1:2^i]
    		result = CatabraFFT.FFT(x)
	end
end

@testset "Radix-3" begin
    	n = 13
    	for i in 1:n
    		x = [ComplexF64(rand(), rand()) for _ in 1:3^i]
    		result = CatabraFFT.FFT(x)
	end
end

@testset "Radix-5" begin
    	n =10
	for i in 1:n
    		x = [ComplexF64(rand(), rand()) for _ in 1:5^i]
    		result = CatabraFFT.FFT(x)
	end
end

@testset "Radix-7" begin
    n = 8
    for i in 1:n
    		x = [ComplexF64(rand(), rand()) for _ in 1:7^i]
    		result = CatabraFFT.FFT(x)
	end
end

@testset "Mixed-Radix" begin
	n = 7
	for i in 1:n
    		x = [ComplexF64(rand(), rand()) for _ in 1:10^i]
		result = CatabraFFT.FFT(x)
	end
end

@testset "Rader's FFT" begin
	n = 1721
	for n in [11,13,17,19, 43, 89, 1721]
    		x = [ComplexF64(rand(), rand()) for _ in 1:n]
		result = CatabraFFT.FFT(x)
	end
end
