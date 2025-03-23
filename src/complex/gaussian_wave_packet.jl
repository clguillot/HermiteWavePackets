
#=
    Represents the complex gaussian function
        λ*exp(-∑ₖ zₖ/2*(xₖ-qₖ)²)*exp(i∑ₖpₖxₖ)
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

function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Tq, Tp}
    return GaussianWavePacket(
        convert(Tλ, G.λ),
        Tz.(G.z),
        Tq.(G.q),
        Tp.(G.p)
    )
end

function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}, G::Gaussian{D}) where{D, Tλ, Tz, Tq, Tp}
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

function truncate_to_gaussian(G::GaussianWavePacket)
    return G
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:GaussianWavePacket}, ::Type{GaussianWavePacket})
    return GaussianWavePacket
end
function Base.promote_rule(::Type{GaussianWavePacket{D, Tλ1, Tz1, Tq1, Tp1}}, ::Type{GaussianWavePacket{D, Tλ2, Tz2, Tq2, Tp2}}) where{D, Tλ1, Tz1, Tq1, Tp1, Tλ2, Tz2, Tq2, Tp2}
    return GaussianWavePacket{D, promote_type(Tλ1, Tλ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

# 
function Base.promote_rule(::Type{<:Gaussian}, ::Type{GaussianWavePacket})
    return GaussianWavePacket
end
function Base.promote_rule(::Type{Gaussian{D, Tλ, Ta, Tq}}, ::Type{TG}) where{D, Tλ, Ta, Tq, TG<:GaussianWavePacket}
    return promote_type(GaussianWavePacket{D, Tλ, Tq, Tq, Tq}, TG)
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
function Base.zero(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}) where{D, Tλ, Tz, Tq, Tp}
    return GaussianWavePacket(zero(Tλ), ones(SVector{D, Tz}), zeros(SVector{D, Tq}), zeros(SVector{D, Tp}))
end

# Creates a copy of a gaussian
function Base.copy(G::GaussianWavePacket)
    return GaussianWavePacket(G.λ, G.z, G.q, G.p)
end

#
function core_type(::Type{GaussianWavePacket{D, Tλ, Tz, Tq, Tp}}) where{D, Tλ, Tz, Tq, Tp}
    return promote_type(Tλ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a gaussian
function Base.conj(G::GaussianWavePacket)
    return GaussianWavePacket(conj.(G.λ), conj.(G.z), G.q, -G.p)
end

# Evaluates a gaussian at x
function (G::GaussianWavePacket{D})(x::AbstractVector{<:Number}) where D
    xs = SVector{D}(x)
    return G.λ * exp(-sum(z/2 * (y - q)^2 for (z, q, y) in zip(G.z, G.q, xs))) * cis(dot(G.p, xs))
end

#=
    TRANSFORMATIONS
=#

# 
function Base.:-(G::GaussianWavePacket)
    return GaussianWavePacket(-G.λ, G.z, G.q, G.p)
end

# Computes the product of a scalar and a gaussian
function Base.:*(w::Number, G::GaussianWavePacket)
    return GaussianWavePacket(w * G.λ, G.z, G.q, G.p)
end

# Computes the product of a gaussian by a scalar
function Base.:/(G::GaussianWavePacket, w::Number)
    return GaussianWavePacket(G.λ / w, G.z, G.q, G.p)
end

# Computes the product of two gaussians
function Base.:*(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    z, q, p = complex_gaussian_product_arg(G1.z, G1.q, G1.p, G2.z, G2.q, G2.p)
    λ = G1(q) * G2(q) * cis(-dot(p, q))
    return GaussianWavePacket(λ, z, q, p)
end

# Multiplies a gaussian wave packet by exp(-i∑ₖbₖ/2 * (xₖ - qₖ)^2) * exp(ipx)
function unitary_product(b::AbstractVector{<:Real}, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, G::GaussianWavePacket{D}) where D
    b = SVector{D}(b)
    q = SVector{D}(q)
    p = SVector{D}(p)
    u = @. b * (G.q + q) * (G.q - q)
    λ_ = G.λ * cis(sum(u) / 2)
    z_ = @. G.z + complex(0, b)
    q_ = G.q
    p_ = @. G.p + p - b * (G.q - q)
    return GaussianWavePacket(λ_, z_, q_, p_)
end
# Multiplies a gaussian wave packet by exp(-ib/2 * x^2)
function unitary_product(b::AbstractVector{<:Real}, G::GaussianWavePacket{D}) where D
    b = SVector{D}(b)
    u = @. b * G.q^2
    λ_ = G.λ * cis(sum(u) / 2)
    z_ = @. G.z + complex(0, b)
    q_ = G.q
    p_ = @. G.p - b * G.q
    return GaussianWavePacket(λ_, z_, q_, p_)
end

# Computes the integral of a gaussian
function integral(G::GaussianWavePacket{D}) where D
    T = fitting_float(G)
    return T((2π)^(D/2)) * G.λ / prod(sqrt.(G.z)) * cis(dot(G.p, G.q)) * exp(-sum(p^2 / (2*z) for (z, p) in zip(G.z, G.p)))
end

#=
    Computes the Fourier transform of a gaussian
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
function fourier(G::GaussianWavePacket{D}) where D
    T = fitting_float(G)
    z_tf, q_tf, p_tf = complex_gaussian_fourier_arg(G.z, G.q, G.p)
    λ_tf = T((2π)^(D/2)) * G.λ / prod(sqrt.(G.z)) * cis(dot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

#=
    Computes the inverse Fourier transform of a gaussian
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻ᴰ∫dξ e^(ixξ) ψ(ξ)
=#
function inv_fourier(G::GaussianWavePacket{D}) where D
    T = fitting_float(G)
    z_tf, q_tf, p_tf = complex_gaussian_inv_fourier_arg(G.z, G.q, G.p)
    λ_tf = T((2π)^(-D/2)) * G.λ / prod(sqrt.(G.z)) * cis(dot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

# Computes the convolution product of two gaussians
function convolution(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    z1, q1, p1 = G1.z, G1.q, G1.p
    λ2, z2, q2, p2 = G2.λ, G2.z, G2.q, G2.p
    z, q, p = complex_gaussian_convolution_product_arg(z1, q1, p1, z2, q2, p2)
    λ = cis(dot(q, p2 - p)) * integral(G1 * GaussianWavePacket(λ2, z2, q - q2, -p2))
    return GaussianWavePacket(λ, z, q, p)
end

# Computes the L² product of two gaussian wave packets
function dot_L2(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    return integral(conj(G1) * G2)
end

# Computes the square L² norm of a gaussian wave packet
function norm2_L2(G::GaussianWavePacket{D}) where D
    T = fitting_float(G)
    return T(sqrt(π)^D) * abs2(G.λ) * prod(real.(G.z))^T(-1/2)
end