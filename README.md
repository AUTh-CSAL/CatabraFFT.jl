# CatabraFFT

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://pitsianis.github.io/CatabraFFT.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://pitsianis.github.io/CatabraFFT.jl/dev/)
[![Build Status](https://github.com/pitsianis/CatabraFFT.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/pitsianis/CatabraFFT.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/pitsianis/CatabraFFT.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/pitsianis/CatabraFFT.jl)



![Alt text](./svgs/radix-2-bench.svg)
![Alt text](./svgs/120_prime_power_on.svg)

CatabraFFT.jl is a tool that auto-generates C2C FFT kernels for any arbitrary size. It makes use of hardware resources to produce FFT kernels calibrated for SIMD-friendly instructions using the metaprogramming features of the Julia compiler.

The performance of CatabraFFT.jl rivals that of FFTW.jl.

The CatabraFFT.jl project, proposes a framework of producing auto-generated high-performance FFT kernels as codelet segments combined together as boosted
functions, all written using only the Julia programming language.

It is separated in two stages; The first stage is pre-complication where, given the signal length, a plan of symbolic instructions is generated. After a symbolic representation of the chosen function is set, called the plan of the function, the plan is converted to actual pre-written code segments that make a part of the final optimal function CatabraFFT.jl designed to use.

Depending on the signal length ```n``` integer, the first criterion of the code creator is whether n belongs to the domain of values ```2^a``` , ```3^b``` ```5^c``` or ```7^d``` . The powers of these single digit primes are accelerated since there are already hand-written codelets of given radix-n size that
can be properly placed to construct unrolled linear functions.

If ```n = 2^a ∗ 3^b ∗ 5^c ∗ 7^d``` then the mixed-radix formulation is applied where input
size n is broken down to the greates common divisor (gcd), ```n = n_1 ∗ n_2``` . This way, since we know that ```n_1 ∪ n_2 ∈
{2^a , 3^b , 5^c , 7^d }``` the family of accelerated codelets fits to the two sub-lengths.

In the case of the set of other non-prime integers, a previously deduced recursive apprach of the mixed-radix formulation is applied, breaking
down the initial ```n``` length down to cached-in precomputed codelet fits, or just prime length sub-problems.

Finally, in the case of prime-sized input signals, we apply Rader’s FFT algorithm for prime signal sizes.

The second stage consists of using all pre-computed assets and create actually competitive benchmark scores with peak  FFT libraries like FFTW.

## Run Code

You can ```git clone``` the current working repo 

or you can install the (upcoming) package with
```Pkg] add CatabraFFT```
```using CatabraFFT```
```y = CatabraFFT.fft(x)```
```x = CatabraFFT.ifft(Y)```

There are bencharking tools in ```/test/runtests.jl``` that pair the benchmarks of CatabraFFT.jl with FFTW.jl.
You can view time plots by installing the VS Code Julia Extention.
