struct U1{T<:AbstractFloat} <: Number
    angle::T

    U1{T}(θ::T) where T<:AbstractFloat = new{T}(mod2pi(θ))
end

# Constructors
U1(θ::T) where T<:AbstractFloat = U1{T}(θ)
U1(z::Complex{T}) where T<:AbstractFloat = U1{T}(angle(z))

# Convert to Complex when needed
Complex{T}(u::U1{T}) where T<:AbstractFloat = Complex{T}(cospi(u.angle/π), sinpi(u.angle/π))
Base.convert(::Type{Complex{T}}, u::U1{T}) where T<:AbstractFloat = Complex{T}(u)

# Basic U1 operations
Base.:*(a::U1{T}, b::U1{T}) where T<:AbstractFloat = U1{T}(a.angle + b.angle)
Base.:/(a::U1{T}, b::U1{T}) where T<:AbstractFloat = U1{T}(a.angle - b.angle)
Base.conj(u::U1{T}) where T<:AbstractFloat = U1{T}(-u.angle)
Base.abs(u::U1{T}) where T<:AbstractFloat = one(T)

# Operations with Complex numbers
# U1 * Complex = Complex
function Base.:*(u::U1{T}, z::Complex{T}) where T<:AbstractFloat
    r = abs(z)
    θ = angle(z)
    Complex{T}(r * cos(u.angle + θ), r * sin(u.angle + θ))
end

# Complex * U1 = Complex
Base.:*(z::Complex{T}, u::U1{T}) where T<:AbstractFloat = u * z

# U1 / Complex = Complex
function Base.:/(u::U1{T}, z::Complex{T}) where T<:AbstractFloat
    r = abs(z)
    θ = angle(z)
    Complex{T}(cos(u.angle - θ) / r, sin(u.angle - θ) / r)
end

# Complex / U1 = Complex
function Base.:/(z::Complex{T}, u::U1{T}) where T<:AbstractFloat
    r = abs(z)
    θ = angle(z)
    Complex{T}(r * cos(θ - u.angle), r * sin(θ - u.angle))
end

# Power operations
Base.:^(u::U1{T}, n::Integer) where T<:AbstractFloat = U1{T}(u.angle * n)
Base.:^(u::U1{T}, x::AbstractFloat) where T<:AbstractFloat = U1{T}(u.angle * x)

# Additional utility functions
Base.angle(u::U1{T}) where T<:AbstractFloat = u.angle
Base.exp(u::U1{T}) where T<:AbstractFloat = Complex{T}(u)
Base.log(u::U1{T}) where T<:AbstractFloat = Complex{T}(zero(T), u.angle)

# Promote rules for mixed operations
Base.promote_rule(::Type{U1{T}}, ::Type{Complex{T}}) where T<:AbstractFloat = Complex{T}
Base.promote_rule(::Type{U1{T}}, ::Type{T}) where T<:AbstractFloat = Complex{T}

# Show method for nice printing
Base.show(io::IO, u::U1) = print(io, "U1($(u.angle))")
