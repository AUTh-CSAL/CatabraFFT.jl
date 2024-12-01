using CatabraFFT, FFTW, Test

@testset "Mixed-Radix" begin
	for n in 10 .^(1:7)
    	x = randn(ComplexF64,n)
		result = CatabraFFT.FFT(x)
        @test result ≈ FFTW.fft(x)
	end
end

@testset "Rader's FFT" begin
	for n in [11,13,17,19, 43, 89, 1721]
        x = randn(ComplexF64,n)
		result = CatabraFFT.FFT(x)
        @test result ≈ FFTW.fft(x)
	end
end

for (b,maxexp) in [(2,20), (3,13), (5,10), (7,8)]
    @testset "Radix-$b" begin
        for n in b.^(1:maxexp)
            x = randn(ComplexF64,n)
            result = CatabraFFT.FFT(x)
            @test result ≈ FFTW.fft(x)
        end
    end
end