using CatabraFFT, FFTW, Test


for (transformFunction, referenceTransform, transformationType) in
    [(CatabraFFT.fft, FFTW.fft, "Forward"), (CatabraFFT.ifft, FFTW.ifft, "Inverse")]
    for CType in [ComplexF64, ComplexF32] #ComplexF16
        @testset "$CType $transformationType FFT" begin

            @testset "Mixed-Radix $transformationType FFT $CType" begin
                for n in 10 .^ (1:7)
                    x = randn(CType, n)
                    result = transformFunction(x)
                    @test result ≈ referenceTransform(x)
                end
            end

            @testset "Rader's $transformationType FFT $CType" begin
                for n in [11, 13, 17, 19, 43, 89, 1721]
                    x = randn(CType, n)
                    result = transformFunction(x)
                    @test result ≈ referenceTransform(x)
                end
            end

            for (b, maxexp) in [(2, 20), (3, 13), (5, 10), (7, 8)]
                @testset "Radix-$b $transformationType FFT $CType" begin
                    for n in b .^ (1:maxexp)
                        x = randn(CType, n)
                        result = transformFunction(x)
                        @test result ≈ referenceTransform(x)
                    end
                end
            end

            println("CLEANING CACHED-IN FUNCTIONS FOR NEW IVDEP FUNCTION")
            CatabraFFT.empty_cache()

            for (b, maxexp) in [(2, 20), (3, 13), (5, 10), (7, 8)]
                @testset "Radix-$b IVDEP $transformationType FFT $CType" begin
                    for n in b .^ (1:maxexp)
                        x = randn(CType, n)
                        result = transformFunction(x, true)
                        @test result ≈ referenceTransform(x)
                    end
                end
            end

        end
    end
end

for CType in [ComplexF64, ComplexF32, ComplexF16]
    @testset "$CType FFT selftest" begin
        for n in [89, 121, 2^10, 2^11, 3^7, 5^6, 10^4]
            x = randn(CType, n)
            @test x ≈ CatabraFFT.ifft(CatabraFFT.fft(x))
        end
    end
end