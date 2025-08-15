# using Pkg

# # Add the package from a specific branch
# Pkg.add(url="https://github.com/JuliaSymbolics/Metatheory.jl", rev="ale/3.0")

using Metatheory
using Metatheory.EGraphs

# root of unity rules
rootofunity = @theory n k q begin
  ω(n)^n --> 1
  ω(n)^(n / 2) --> -1
  ω(n)^(n / 4) --> im
  ω(n)^(3n / 4) --> -im
  ω(k * n)^q --> ω(n)^(q / k)
end

expr = :(ω(2n)^(n))
rewrite(expr, rootofunity)


tol_scale = 10.0
zerosones = @theory x begin
  x::(xs -> xs isa AbstractFloat && abs(xs) < tol_scale * eps(typeof(xs))) --> 0
  x::(xs -> xs isa AbstractFloat && abs(xs - 1) < tol_scale * eps(typeof(xs))) --> sign(x) * 1
  0 * x --> 0
  # x * 0 --> 0
  0 + x --> x
  # x + 0 --> x
  x - 0 --> x
  0 - x --> -x
  1 * x --> x
  x * 1 --> x
  x / 1 --> x
  -1 * x --> -x
end

rewrite(:((1e-16 * x[1] + (-1.00000000000000001 - 1e-16im) * x[2])^2), zerosones)

# expand DFT rules
expanddft = @theory n nn begin
  F(2(n)) --> kron(F(2), I(n)) * D(2, n) * kron(I(2), F(n)) * P(2n, 2)
end
rewrite(:(F(2 * 2)), expanddft)


# ω(n) = exp.(-2 * pi * im / n)
# expandfft = @theory n x begin
#   F(1) * x --> x
#   F(2) * x --> [x[1] + x[2], x[1] - x[2]]
#   F(n) * x --> quote
#     let n2 = 2
#       n2
#     end
#   end
# end

# rewrite(:(F(4) * x),expandfft) 
# #   let n2 = n ÷ 2, w = ω(n) .^ (0:n2-1)
# #     let y1 = [sum(k -> ω(n2)^(j * k) * x[2k+1], 0:n2-1) for j = 0:n2-1],
# #       y2 = w .* [sum(k -> ω(n2)^(j * k) * x[2k+1+1], 0:n2-1) for j = 0:n2-1]

# #       vcat(y1 + y2,
# #         y1 - y2)
# #     end
# #   end
# # end


