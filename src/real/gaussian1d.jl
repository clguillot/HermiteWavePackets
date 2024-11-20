import Base.*
import Base.copy
import Base.zero

export Gaussian1D
export integral
export convolution
export dot_L2

#=
    Represents the gaussian function
        λ*exp(-a/2*(x-q)²)
=#
struct Gaussian1D{Tλ<:Real, Ta<:Real, Tq<:Real}
    λ::Tλ
    a::Ta
    q::Tq
end

#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
@inline function zero(::Type{Gaussian1D{Tλ, Ta, Tq}}) where{Tλ, Ta, Tq}
    return Gaussian1D(zero(Tλ), one(Ta), zero(Tq))
end

# Creates a copy of a gaussian
@inline function copy(G::Gaussian1D)
    return Gaussian1D(G.λ, G.a, G.q)
end

# Evaluates a gaussian at x
@inline function (G::Gaussian1D)(x::Number)
    return G.λ * myexp(-G.a/2 * (x - G.q)^2)
end

#=
    TRANSFORMATIONS
=#

#=
    Computes a, q such that
        exp(-a/2*(x-q)²) = C * exp(-a1/2*(x-q1)²) * exp(-a2/2*(x-q2)²)
    where C is some constant
=#
@inline function gaussian_product_arg(a1::Real, q1::Real, a2::Real, q2::Real)
    a = a1 + a2
    q = (a1 * q1 + a2 * q2) / (a1 + a2)
    return a, q
end

#=
    Computes a, q such that exp(-a/2*(x-q)²) is equal (up to some constant)
    to the convolution product
        exp(-a1/2*(x-q1)²) ∗ exp(-a2/2*(x-q2)²)
=#
@inline function gaussian_convolution_arg(a1::Real, q1::Real, a2::Real, q2::Real)
    a = a1 * a2 / (a1 + a2)
    q = q1 + q2
    return a, q
end

# Computes the product of a scalar and a gaussian
@inline function (*)(μ::Real, G::Gaussian1D)
    return Gaussian1D(μ * G.λ, G.a, G.q)
end

# Computes the product of two gaussians
@inline function (*)(G1::Gaussian1D, G2::Gaussian1D)
    λ1, a1, q1 = G1.λ, G1.a, G1.q
    λ2, a2, q2 = G2.λ, G2.a, G2.q
    a, q = gaussian_product_arg(a1, q1, a2, q2)
    λ = λ1 * λ2 * myexp(- a1*(q-q1)^2 / 2) * myexp(- a2*(q-q2)^2 / 2)
    return Gaussian1D(λ, a, q)
end

# Computes the integral of a gaussian
@inline function integral(G::Gaussian1D)
    return G.λ * sqrt(2 * (π / G.a))
end

# Computes the convolution product of two gaussians
@inline function convolution(G1::Gaussian1D, G2::Gaussian1D)
    λ1, a1, q1 = G1.λ, G1.a, G1.q
    λ2, a2, q2 = G2.λ, G2.a, G2.q
    a, q = gaussian_convolution_arg(a1, q1, a2, q2)
    λ = integral(Gaussian1D(λ1, a1, q1) * Gaussian1D(λ2, a2, q - q2))
    return Gaussian1D(λ, a, q)
end

# Computes the L² product of two gaussians
@inline function dot_L2(G1::Gaussian1D, G2::Gaussian1D)
    return integral(G1 * G2)
end