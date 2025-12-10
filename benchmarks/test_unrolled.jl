using CatabraFFT, FFTW, BenchmarkTools, LinearAlgebra

function tester(T, n)
  x = rand(T, n)

  F = FFTW.plan_fft(x; flags=FFTW.EXHAUSTIVE)
  C = CatabraFFT.plan_fft(x, CatabraFFT.ENCHANT)

  print("Catabra :")
  @btime $C * y setup = (y = rand($T, $n))
  print("FFTW    :")
  @btime $F * y setup = (y = rand($T, $n))
  return norm(C * x - F * x,Inf)
end

for T in [ComplexF32]
  for n = 2 .^ (1:5)
    err = tester(T, n)
    println("$T $n $err\n")
  end
end