
#=
    Represents the gaussian function
        λ*exp(-∑ₖ aₖ/2*(xₖ-qₖ)²)
=#
struct Gaussian{D, Tλ<:Number, Ta<:Real, Tq<:Real} <: AbstractWavePacket
    λ::Tλ
    a::SVector{D, Ta}
    q::SVector{D, Tq}
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{Gaussian{D, Tλ, Ta, Tq}}, G::Gaussian{D}) where {D, Tλ, Ta, Tq}
    return Gaussian(
        Tλ(G.λ),
        Ta.(G.a),
        Tq.(G.q)
    )
end

function Base.convert(::Type{Gaussian{1, Tλ, Ta, Tq}}, G::Gaussian1D) where {Tλ, Ta, Tq}
    return Gaussian(
        Tλ(G.λ),
        SVector{1}(G.a),
        SVector{1}(G.q)
    )
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:Gaussian}, ::Type{Gaussian})
    return Gaussian
end
function Base.promote_rule(::Type{Gaussian{D, Tλ1, Ta1, Tq1}}, ::Type{Gaussian{D, Tλ2, Ta2, Tq2}}) where{D, Tλ1, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return Gaussian{D, promote_type(Tλ1, Tλ2), promote_type(Ta1, Ta2), promote_type(Tq1, Tq2)}
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
function zero(::Type{Gaussian{D, Tλ, Ta, Tq}}) where{D, Tλ, Ta, Tq}
    return Gaussian(zero(Tλ), (@SVector ones(Ta, D)), (@SVector zeros(Tq, D)))
end

# Creates a copy of a gaussian
function copy(G::Gaussian)
    return Gaussian(G.λ, G.a, G.q)
end

#
function core_type(::Type{Gaussian{D, Tλ, Ta, Tq}}) where{D, Tλ, Ta, Tq}
    return promote_type(Tλ, Ta, Tq)
end

# 
function fitting_float(::Type{Gaussian{D, Tλ, Ta, Tq}}) where{D, Tλ, Ta, Tq}
    return fitting_float(promote_type(Tλ, Ta, Tq))
end

# Returns the complex conjugate of a gaussian
function conj(G::Gaussian)
    return Gaussian(conj.(G.λ), G.a, G.q)
end

# Evaluates a gaussian at x
function (G::Gaussian{D})(x::AbstractVector{<:Number}) where{D}
    xs = SVector{D}(x)
    u = @. G.a * (xs - G.q)^2
    return G.λ * exp(-sum(u) / 2)
end

# Evaluates a gaussian at every point in x
function evaluate(G::Gaussian{D}, x::SMatrix{D, M, <:Number}) where{D, M}
    u = @. G.a * (x - G.q)^2
    v = reshape(sum(u; dim=1), M)
    return @. G.λ * exp(-v / 2)
end

#=
    TRANSFORMATIONS
=#

# 
function Base.:-(G::Gaussian)
    return Gaussian(-G.λ, G.a, G.q)
end

# Computes the product of a scalar and a gaussian
function (*)(w::Number, G::Gaussian)
    return Gaussian(w * G.λ, G.a, G.q)
end

# Computes the product of two gaussians
function (*)(G1::Gaussian{D}, G2::Gaussian{D}) where D
    a, q = gaussian_product_arg(G1.a, G1.q, G2.a, G2.q)
    λ = G1(q) * G2(q)
    return Gaussian(λ, a, q)
end

# Computes the integral of a gaussian
function integral(G::Gaussian{D}) where D
    T = fitting_float(G)
    u = @. T(sqrt(2π)) * G.a^T(-1/2)
    return G.λ * prod(u)
end

# Computes the convolution product of two gaussians
function convolution(G1::Gaussian{D}, G2::Gaussian{D}) where D
    a, q = gaussian_convolution_arg(G1.a, G1.q, G2.a, G2.q)
    λ = integral(G1 * Gaussian(G2.λ, G2.a, q - G2.q))
    return Gaussian(λ, a, q)
end

# Computes the L² product of two gaussians
function dot_L2(G1::Gaussian{D, Tλ1, Ta1, Tq1}, G2::Gaussian{D, Tλ2, Ta2, Tq2}) where{D, Tλ1<:Real, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return integral(G1 * G2)
end
function dot_L2(G1::Gaussian{D, Tλ1, Ta1, Tq1}, G2::Gaussian{D, Tλ2, Ta2, Tq2}) where{D, Tλ1, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return integral(conj(G1) * G2)
end

# Computes the square L² norm of a gaussian
function norm2_L2(G::Gaussian{D}) where D
    T = fitting_float(G)
    u = @. T(sqrt(π)) * G.a^T(-1/2)
    return abs2(G.λ) * prod(u)
end