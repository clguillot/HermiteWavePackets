
#=
    Represents the complex gaussian function
        λ*exp(-z/2*(x-q)²)*exp(ipx)
=#
struct GaussianWavePacket{D, Tλ<:Number, Tz<:Number, Tq<:Real, Tp<:Real} <: AbstractWavePacket
    λ::Tλ
    z::SVector{D, Tz}
    q::SVector{D, Tq}
    p::SVector{D, Tp}
end

#=
    CONVERSIONS
=#

function convert(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Tq, Tp}
    return GaussianWavePacket(
        convert(Tλ, G.λ),
        Tz.(G.z),
        Tq.(G.q),
        Tp.(G.p)
    )
end

function convert(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}, G::Gaussian{D}) where{D, Tλ, Tz, Tq, Tp}
    return GaussianWavePacket(
        convert(Tλ, G.λ),
        Tz.(G.a),
        Tq.(G.q),
        zero(SVector{D, Tp})
    )
end

function GaussianWavePacket(G::Gaussian{D, Tλ, Ta, Tq}) where{D, Tλ, Ta, Tq}
    return convert(GaussianWavePacket{D, Tλ, Ta, Tq, Tq}, G)
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:GaussianWavePacket}, ::Type{GaussianWavePacket})
    return GaussianWavePacket1D
end
function Base.promote_rule(::Type{GaussianWavePacket{D, Tλ1, Tz1, Tq1, Tp1}}, ::Type{GaussianWavePacket{D, Tλ2, Tz2, Tq2, Tp2}}) where{D, Tλ1, Tz1, Tq1, Tp1, Tλ2, Tz2, Tq2, Tp2}
    return GaussianWavePacket{D, promote_type(Tλ1, Tλ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

# 
function Base.promote_rule(::Type{<:Gaussian}, ::Type{GaussianWavePacket})
    return GaussianWavePacket1D
end
function Base.promote_rule(::Type{Gaussian{D, Tλ, Ta, Tq}}, ::Type{TG}) where{D, Tλ, Ta, Tq, TG<:GaussianWavePacket1D}
    return promote_type(GaussianWavePacket{D, Tλ, Tq, Tq, Tq}, TG)
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
function zero(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}) where{D, Tλ, Tz, Tq, Tp}
    return GaussianWavePacket(zero(Tλ), (@SVector ones(Tz, D)), (@SVector zeros(Tq, D)), (@SVector zeros(Tp, D)))
end

# Creates a copy of a gaussian
function copy(G::GaussianWavePacket)
    return GaussianWavePacket1D(G.λ, G.z, G.q, G.p)
end

#
function core_type(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}) where{D, Tλ, Tz, Tq, Tp}
    return promote_type(Tλ, Tz, Tq, Tp)
end

# 
function fitting_float(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}) where{D, Tλ, Tz, Tq, Tp}
    return fitting_float(promote_type(Tλ, Tz, Tq, Tp))
end

# Returns the complex conjugate of a gaussian
function conj(G::GaussianWavePacket)
    return GaussianWavePacket(conj(G.λ), conj(G.z), G.q, -G.p)
end

# Evaluates a gaussian at x
function (G::GaussianWavePacket{D})(x::AbstractVector{Number}) where{D}
    xs = SVector{D}(x)
    u = @. exp(-G.z/2 * (xs - G.q)^2) * cis(G.p * xs)
    return G.λ * prod(u)
end

#=
    TRANSFORMATIONS
=#

# Computes the product of a scalar and a gaussian
function (*)(w::Number, G::GaussianWavePacket)
    return GaussianWavePacket1D(w * G.λ, G.z, G.q, G.p)
end

# Computes the product of two gaussians
function (*)(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where{D}
    z, q, p = complex_gaussian_product_arg(G1.z, G1.q, G1.p, G2.z, G2.q, G2.p)
    λ = G1(q) * G2(q) * cis(-p*q)
    return GaussianWavePacket(λ, z, q, p)
end

# # Multiplies a gaussian wave packet by exp(-ib/2 * (x - q)^2) * exp(ipx)
# function unitary_product(b::Real, q::Real, p::Real, G::GaussianWavePacket1D)
#     α = cis(b / 2 * (G.q + q) * (G.q - q))
#     return GaussianWavePacket1D(α .* G.λ, G.z + complex(0, b), G.q, G.p + p - b * (G.q - q))
# end
# # Multiplies a gaussian wave packet by exp(-ib/2 * x^2)
# function unitary_product(b::Real, G::GaussianWavePacket1D{Tλ, Tz, Tq, Tp}) where{Tλ, Tz, Tq, Tp}
#     α = cis(b / 2 * G.q^2)
#     return GaussianWavePacket1D(α .* G.λ, G.z + complex(0, b), G.q, G.p - b * G.q)
# end

# Computes the integral of a gaussian
function integral(G::GaussianWavePacket)
    T = fitting_float(G)
    u = @. T(sqrt(2π)) / sqrt(G.z) * cis(G.p * G.q) * exp(- G.p^2 / (2*G.z))
    return G.λ * prod(u)
end

#=
    Computes the Fourier transform of a gaussian
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
function fourier(G::GaussianWavePacket)
    T = fitting_float(G)
    z_tf, q_tf, p_tf = complex_gaussian_fourier_arg(G.z, G.q, G.p)
    u = @. cis(G.p*G.q) * T(sqrt(2π)) / sqrt(G.z)
    λ_tf = G.λ * prod(u)
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

#=
    Computes the inverse Fourier transform of a gaussian
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻¹∫dξ e^(ixξ) ψ(ξ)
=#
function inv_fourier(G::GaussianWavePacket)
    T = fitting_float(G)
    z_tf, q_tf, p_tf = complex_gaussian_inv_fourier_arg(G.z, G.q, G.p)
    u = @. T((2π)^(-1/2)) * cis(G.p*G.q) / sqrt(G.z)
    λ_tf = λ * prod(u)
    return GaussianWavePacket1D(λ_tf, z_tf, q_tf, p_tf)
end

# Computes the convolution product of two gaussians
function convolution(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where{D}
    z1, q1, p1 = G1.z, G1.q, G1.p
    λ2, z2, q2, p2 = G2.λ, G2.z, G2.q, G2.p
    z, q, p = complex_gaussian_convolution_product_arg(z1, q1, p1, z2, q2, p2)
    λ = cis(q * (p2 - p)) * integral(G1 * GaussianWavePacket1D(λ2, z2, q - q2, -p2))
    return GaussianWavePacket1D(λ, z, q, p)
end

# Computes the L² product of two gaussian wave packets
function dot_L2(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where{D}
    return integral(conj(G1) * G2)
end

# Computes the square L² norm of a gaussian wave packet
function norm2_L2(G::GaussianWavePacket1D)
    T = fitting_float(G)
    u = @. T(sqrt(π)) * real(G.z)^T(-1/2)
    return abs2(G.λ) * prod(u)
end