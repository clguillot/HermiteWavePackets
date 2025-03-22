
#=
    Represents the gaussian function
        λ*exp(-a/2*(x-q)²)
=#
struct Gaussian1D{Tλ<:Number, Ta<:Real, Tq<:Real} <: AbstractWavePacket1D
    λ::Tλ
    a::Ta
    q::Tq
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{Gaussian1D{Tλ, Ta, Tq}}, G::Gaussian1D) where {Tλ, Ta, Tq}
    return Gaussian1D(
        convert(Tλ, G.λ),
        convert(Ta, G.a),
        convert(Tq, G.q)
    )
end

function truncate_to_gaussian(G::Gaussian1D)
    return G
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:Gaussian1D}, ::Type{Gaussian1D})
    return Gaussian1D
end
function Base.promote_rule(::Type{Gaussian1D{Tλ1, Ta1, Tq1}}, ::Type{Gaussian1D{Tλ2, Ta2, Tq2}}) where{Tλ1, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return Gaussian1D{promote_type(Tλ1, Tλ2), promote_type(Ta1, Ta2), promote_type(Tq1, Tq2)}
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
@inline function Base.zero(::Type{Gaussian1D{Tλ, Ta, Tq}}) where{Tλ, Ta, Tq}
    return Gaussian1D(zero(Tλ), one(Ta), zero(Tq))
end

# Creates a copy of a gaussian
@inline function Base.copy(G::Gaussian1D{Tλ, Ta, Tq}) where{Tλ, Ta, Tq}
    return Gaussian1D(G.λ, G.a, G.q)
end

#
function core_type(::Type{Gaussian1D{Tλ, Ta, Tq}}) where{Tλ, Ta, Tq}
    return promote_type(Tλ, Ta, Tq)
end

# Returns the complex conjugate of a gaussian
@inline function Base.conj(G::Gaussian1D)
    return Gaussian1D(conj(G.λ), G.a, G.q)
end

# Evaluates a gaussian at x
@inline function (G::Gaussian1D)(x::Number)
    return G.λ * exp(-G.a/2 * (x - G.q)^2)
end

#=
    TRANSFORMATIONS
=#

#=
    Computes a, q such that
        exp(-a/2*(x-q)²) = C * exp(-a1/2*(x-q1)²) * exp(-a2/2*(x-q2)²)
    where C is some constant
=#
@inline function gaussian_product_arg(a1, q1, a2, q2)
    a = @. a1 + a2
    q = @. (a1 * q1 + a2 * q2) / (a1 + a2)
    return a, q
end

#=
    Computes a, q such that exp(-a/2*(x-q)²) is equal (up to some constant)
    to the convolution product
        exp(-a1/2*(x-q1)²) ∗ exp(-a2/2*(x-q2)²)
=#
@inline function gaussian_convolution_arg(a1, q1, a2, q2)
    a = @. a1 * a2 / (a1 + a2)
    q = @. q1 + q2
    return a, q
end

# 
@inline function Base.:-(G::Gaussian1D)
    return Gaussian1D(-G.λ, G.a, G.q)
end

# Computes the product of a scalar and a gaussian
@inline function Base.:*(w::Number, G::Gaussian1D)
    return Gaussian1D(w * G.λ, G.a, G.q)
end

# Computes the product of a gaussian by a scalar
function Base.:/(G::Gaussian1D, w::Number)
    return Gaussian1D(G.λ / w, G.a, G.q)
end

# Computes the product of two gaussians
@inline function Base.:*(G1::Gaussian1D, G2::Gaussian1D)
    a, q = gaussian_product_arg(G1.a, G1.q, G2.a, G2.q)
    λ = G1(q) * G2(q)
    return Gaussian1D(λ, a, q)
end

# Computes the integral of a gaussian
@inline function integral(G::Gaussian1D)
    T = fitting_float(G)
    return G.λ * T(sqrt(2π)) * G.a^T(-1/2)
end

# Computes the convolution product of two gaussians
@inline function convolution(G1::Gaussian1D, G2::Gaussian1D)
    a1, q1 = G1.a, G1.q
    λ2, a2, q2 = G2.λ, G2.a, G2.q
    a, q = gaussian_convolution_arg(a1, q1, a2, q2)
    λ = integral(G1 * Gaussian1D(λ2, a2, q - q2))
    return Gaussian1D(λ, a, q)
end

# Computes the L² product of two gaussians
@inline function dot_L2(G1::Gaussian1D{Tλ1, Ta1, Tq1}, G2::Gaussian1D{Tλ2, Ta2, Tq2}) where{Tλ1<:Real, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return integral(G1 * G2)
end
@inline function dot_L2(G1::Gaussian1D{Tλ1, Ta1, Tq1}, G2::Gaussian1D{Tλ2, Ta2, Tq2}) where{Tλ1, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return integral(conj(G1) * G2)
end

# Computes the square L² norm of a gaussian
@inline function norm2_L2(G::Gaussian1D)
    T = fitting_float(G)
    return abs2(G.λ) * T(sqrt(π)) * G.a^T(-1/2)
end