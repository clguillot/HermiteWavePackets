
#=
    Represents the complex gaussian function
        λ*exp(-z/2*(x-q)²)*exp(ipx)
=#
struct GaussianWavePacket1D{Tλ<:Number, Tz<:Number, Tq<:Real, Tp<:Real} <: AbstractWavePacket1D
    λ::Tλ
    z::Tz
    q::Tq
    p::Tp
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{GaussianWavePacket1D{Tλ, Tz, Tq, Tp}}, G::GaussianWavePacket1D) where {Tλ, Tz, Tq, Tp}
    return GaussianWavePacket1D(
        convert(Tλ, G.λ),
        convert(Tz, G.z),
        convert(Tq, G.q),
        convert(Tp, G.p)
    )
end

function Base.convert(::Type{GaussianWavePacket1D{Tλ, Tz, Tq, Tp}}, G::Gaussian1D) where {Tλ, Tz, Tq, Tp}
    return GaussianWavePacket1D(
        convert(Tλ, G.λ),
        convert(Tz, G.a),
        convert(Tq, G.q),
        zero(Tp)
    )
end

function GaussianWavePacket1D(G::Gaussian1D{Tλ, Ta, Tq}) where {Tλ, Ta, Tq}
    return convert(GaussianWavePacket1D{Tλ, Ta, Tq, Tq}, G)
end

function truncate_to_gaussian(G::GaussianWavePacket1D)
    return G
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:GaussianWavePacket1D}, ::Type{GaussianWavePacket1D})
    return GaussianWavePacket1D
end
function Base.promote_rule(::Type{GaussianWavePacket1D{Tλ1, Tz1, Tq1, Tp1}}, ::Type{GaussianWavePacket1D{Tλ2, Tz2, Tq2, Tp2}}) where{Tλ1, Tz1, Tq1, Tp1, Tλ2, Tz2, Tq2, Tp2}
    return GaussianWavePacket1D{promote_type(Tλ1, Tλ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

# 
function Base.promote_rule(::Type{<:Gaussian1D}, ::Type{GaussianWavePacket1D})
    return GaussianWavePacket1D
end
function Base.promote_rule(::Type{Gaussian1D{Tλ, Ta, Tq}}, ::Type{TG}) where{Tλ, Ta, Tq, TG<:GaussianWavePacket1D}
    return promote_type(GaussianWavePacket1D{Tλ, Tq, Tq, Tq}, TG)
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
@inline function Base.zero(::Type{GaussianWavePacket1D{Tλ, Tz, Tq, Tp}}) where{Tλ, Tz, Tq, Tp}
    return GaussianWavePacket1D(zero(Tλ), one(Tz), zero(Tq), zero(Tp))
end

# Creates a copy of a gaussian
@inline function Base.copy(G::GaussianWavePacket1D)
    return GaussianWavePacket1D(G.λ, G.z, G.q, G.p)
end

#
function core_type(::Type{GaussianWavePacket1D{Tλ, Tz, Tq, Tp}}) where{Tλ, Tz, Tq, Tp}
    return promote_type(Tλ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a gaussian
@inline function Base.conj(G::GaussianWavePacket1D)
    return GaussianWavePacket1D(conj(G.λ), conj(G.z), G.q, -G.p)
end

# Evaluates a gaussian at x
@inline function (G::GaussianWavePacket1D)(x::Number)
    return G.λ * exp(-G.z/2 * (x - G.q)^2) * cis(G.p * x)
end

#=
    TRANSFORMATIONS
=#

#=
    Computes z, q, p such that exp(-z/2*(x-q)²)*exp(ipx) is equal
    to the product
        C * exp(-z1/2*(x-q1)²)*exp(ip1*x) * exp(-z2/2*(x-q2)²)*exp(ip2*x)
    where C is some constant
=#
@inline function complex_gaussian_product_arg(z1, q1, p1, z2, q2, p2)
    z = @. z1 + z2
    q = @. (real(z1) * q1 + real(z2) * q2) / (real(z1) + real(z2))
    p0 = @. (imag(z1) * q1 + imag(z2) * q2) - (imag(z1) + imag(z2)) * q
    p = @. p2 + p1 + p0
    return z, q, p
end

#=
    Computes z_tf, q_tf, p_tf such that exp(-z_tf/2*(x-q_tf)²)*exp(ip_tf*x) is equal
    to the Fourier transform of
        C * exp(-z/2*(x-q)²)*exp(ip*x)
    where C is some constant
=#
@inline function complex_gaussian_fourier_arg(z, q, p)
    return @. inv(z), p, -q
end

#=
    Computes z_itf, q_itf, p_itf such that exp(-z_itf/2*(x-q_itf)²)*exp(ip_itf*x) is equal
    to the inverse Fourier transform of
        C * exp(-z/2*(x-q)²)*exp(ip*x)
    where C is some constant
=#
@inline function complex_gaussian_inv_fourier_arg(z, q, p)
    return @. inv(z), -p, q
end

#=
    Computes z, q, p such that exp(-z/2*(x-q)²)*exp(ipx) is equal
    to the convolution product
        C * exp(-z1/2*(x-q1)²)*exp(ip1*x) ∗ exp(-z2/2*(x-q2)²)*exp(ip2*x)
    where C is some constant
=#
@inline function complex_gaussian_convolution_product_arg(z1, q1, p1, z2, q2, p2)
    z1_tf, q1_tf, p1_tf = complex_gaussian_fourier_arg(z1, q1, p1)
    z2_tf, q2_tf, p2_tf = complex_gaussian_fourier_arg(z2, q2, p2)
    z_tf, q_tf, p_tf = complex_gaussian_product_arg(z1_tf, q1_tf, p1_tf, z2_tf, q2_tf, p2_tf)
    return complex_gaussian_inv_fourier_arg(z_tf, q_tf, p_tf)
end

# 
@inline function Base.:-(G::GaussianWavePacket1D)
    return GaussianWavePacket1D(-G.λ, G.z, G.q, G.p)
end

# Computes the product of a scalar and a gaussian
@inline function Base.:*(w::Number, G::GaussianWavePacket1D)
    return GaussianWavePacket1D(w * G.λ, G.z, G.q, G.p)
end

# Computes the product of a gaussian by a scalar
@inline function Base.:/(G::GaussianWavePacket1D, w::Number)
    return GaussianWavePacket1D(G.λ / w, G.z, G.q, G.p)
end

# Computes the product of two gaussians
@inline function Base.:*(G1::GaussianWavePacket1D, G2::GaussianWavePacket1D)
    z, q, p = complex_gaussian_product_arg(G1.z, G1.q, G1.p, G2.z, G2.q, G2.p)
    λ = G1(q) * G2(q) * cis(-p*q)
    return GaussianWavePacket1D(λ, z, q, p)
end

# Multiplies a gaussian wave packet by exp(-ib/2 * (x - q)^2) * exp(ipx)
@inline function unitary_product(b::Real, q::Real, p::Real, G::GaussianWavePacket1D)
    α = cis(b / 2 * (G.q + q) * (G.q - q))
    return GaussianWavePacket1D(α .* G.λ, G.z + complex(0, b), G.q, G.p + p - b * (G.q - q))
end
# Multiplies a gaussian wave packet by exp(-ib/2 * x^2)
@inline function unitary_product(b::Real, G::GaussianWavePacket1D)
    α = cis(b / 2 * G.q^2)
    return GaussianWavePacket1D(α .* G.λ, G.z + complex(0, b), G.q, G.p - b * G.q)
end

# Computes the integral of a gaussian
@inline function integral(G::GaussianWavePacket1D)
    T = fitting_float(G)
    return G.λ * T(sqrt(2π)) / sqrt(G.z) * cis(G.p * G.q) * exp(- G.p^2 / (2*G.z))
end

#=
    Computes the Fourier transform of a gaussian
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
function fourier(G::GaussianWavePacket1D)
    T = fitting_float(G)
    λ, z, q, p = G.λ, G.z, G.q, G.p
    z_tf, q_tf, p_tf = complex_gaussian_fourier_arg(G.z, G.q, G.p)
    λ_tf = λ * cis(p*q) * T(sqrt(2π)) / sqrt(z)
    return GaussianWavePacket1D(λ_tf, z_tf, q_tf, p_tf)
end

#=
    Computes the inverse Fourier transform of a gaussian
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻¹∫dξ e^(ixξ) ψ(ξ)
=#
function inv_fourier(G::GaussianWavePacket1D)
    T = fitting_float(G)
    λ, z, q, p = G.λ, G.z, G.q, G.p
    z_tf, q_tf, p_tf = complex_gaussian_inv_fourier_arg(G.z, G.q, G.p)
    λ_tf = λ * T((2π)^(-1/2)) * cis(p*q) / sqrt(z)
    return GaussianWavePacket1D(λ_tf, z_tf, q_tf, p_tf)
end

# Computes the convolution product of two gaussians
function convolution(G1::GaussianWavePacket1D, G2::GaussianWavePacket1D)
    z1, q1, p1 = G1.z, G1.q, G1.p
    λ2, z2, q2, p2 = G2.λ, G2.z, G2.q, G2.p
    z, q, p = complex_gaussian_convolution_product_arg(z1, q1, p1, z2, q2, p2)
    λ = cis(q * (p2 - p)) * integral(G1 * GaussianWavePacket1D(λ2, z2, q - q2, -p2))
    return GaussianWavePacket1D(λ, z, q, p)
end

# Computes the L² product of two gaussian wave packets
@inline function dot_L2(G1::GaussianWavePacket1D, G2::GaussianWavePacket1D)
    return integral(conj(G1) * G2)
end

# Computes the square L² norm of a gaussian wave packet
@inline function norm2_L2(G::GaussianWavePacket1D)
    T = fitting_float(G)
    return abs2(G.λ) * T(sqrt(π)) * real(G.z)^T(-1/2)
end