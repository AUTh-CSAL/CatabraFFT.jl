using CatabraFFT, FFTW, Test

for CType in [ComplexF64, ComplexF32]
    @testset "Mixed-Radix $CType" begin
        for n in 10 .^ (1:7)
            x = randn(CType, n)
            result = CatabraFFT.FFT(x)
            @test result ≈ FFTW.fft(x)
        end
    end

    @testset "Rader's FFT $CType" begin
        for n in [11, 13, 17, 19, 43, 89, 1721]
            x = randn(CType, n)
            result = CatabraFFT.FFT(x)
            @test result ≈ FFTW.fft(x)
        end
    end

    for (b, maxexp) in [(2, 20), (3, 13), (5, 10), (7, 8)]
        @testset "Radix-$b $CType" begin
            for n in b .^ (1:maxexp)
                x = randn(CType, n)
                result = CatabraFFT.FFT(x)
                @test result ≈ FFTW.fft(x)
            end
        end
    end

end
