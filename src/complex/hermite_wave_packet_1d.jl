
#=
    Reminder :
    -gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,p) such that the quadrature formula
        ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1

    We have the folowing properties (cf https://mathworld.wolfram.com/HermitePolynomial.html)
        ∫dx Hₘ(x)Hₙ(x)exp(-x²) = δₘₙ2ⁿn!√π
        Hₙ₊₁(x) = 2xHₙ(x) - 2nHₙ₋₁(x)
        Hₙ'(x) = 2n Hₙ₋₁(x)

    Now define for z = a + ib, a > 0, and p, q real numbers
        ψₙ(z, q, p, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*x) * exp(-z(x - q)²/2) * exp(ipx)
    (we shall only write ψₙ(x) when the context is clear enough)
    then it follows from the previous properties that
        ∫̄ψₘψₙ = δₘₙ
        ψₙ₊₁ = (√(2a)*x*ψₙ - √n*ψₙ₋₁) / √(n+1) (where a = real(z))
=#

#=
    Represents the complex hermite function
        ∑ₙ Λ[n+1] ψₙ(z, q, p, x) (n=0,...,N-1)
=#
struct HermiteWavePacket1D{N, TΛ<:Number, Tz<:Number, Tq<:Real, Tp<:Real} <: AbstractWavePacket1D
    Λ::SVector{N, TΛ}
    z::Tz
    q::Tq
    p::Tp
end

#=
    CONVERSIONS
=#

function convert(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}, G::Gaussian1D) where{N, TΛ, Tz, Tq, Tp}
    T = fitting_float(G)
    Λ = [convert(TΛ, (G.a / π)^T(-1/4) * G.λ); zero(SVector{N - 1, TΛ})]
    return HermiteWavePacket1D(Λ, convert(Tz, G.a), convert(Tq, G.q), zero(Tp))
end

function convert(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}, G::GaussianWavePacket1D) where{N, TΛ, Tz, Tq, Tp}
    T = fitting_float(G)
    a = real(G.z)
    Λ = [convert(TΛ, (a / π)^T(-1/4) * G.λ); zero(SVector{N - 1, TΛ})]
    return HermiteWavePacket1D(Λ, convert(Tz, G.z), convert(Tq, G.q),convert(Tp, G.p))
end

function convert(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}, H::HermiteFct1D{N2}) where{N, TΛ, Tz, Tq, Tp, N2}
    if N < N2
        throw(InexactError("Cannot convert HermiteFct1D{N2} to HermiteWavePacket1D{N}: N = $N is smaller than N2 = $N2. Ensure that N ≥ N2 for a valid conversion."))
    end

    Λ = [TΛ.(H.Λ) ; zero(SVector{N - N2, TΛ})]
    return HermiteWavePacket1D(Λ, Tz(H.a), Tq(H.q), zero(Tp))
end

function HermiteWavePacket1D(G::Gaussian1D{Tλ, Ta, Tq}) where{Tλ, Ta, Tq}
    return convert(HermiteWavePacket1D{1, Tλ, Ta, Tq, Tq}, G)
end

function HermiteWavePacket1D(G::GaussianWavePacket1D{Tλ, Tz, Tq, Tp}) where{Tλ, Tz, Tq, Tp}
    return convert(HermiteWavePacket1D{1, Tλ, Tz, Tq, Tp}, G)
end

function HermiteWavePacket1D(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return convert(HermiteWavePacket1D{N, TΛ, Ta, Tq, Tq}, H)
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:HermiteWavePacket1D}, ::Type{HermiteWavePacket1D})
    return HermiteWavePacket1D
end
function Base.promote_rule(::Type{HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}}, ::Type{HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return HermiteWavePacket1D{max(N1, N2), promote_type(TΛ1, TΛ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

# 
function Base.promote_rule(::Type{<:Gaussian1D}, ::Type{HermiteWavePacket1D})
    return HermiteWavePacket1D
end
function Base.promote_rule(::Type{Gaussian1D{Tλ, Ta, Tq}}, ::Type{TH}) where{Tλ, Ta, Tq, TH<:HermiteWavePacket1D}
    return promote_type(HermiteWavePacket1D{1, Tλ, Ta, Tq, Tq}, TH)
end

# 
function Base.promote_rule(::Type{<:GaussianWavePacket1D}, ::Type{HermiteWavePacket1D})
    return HermiteWavePacket1D
end
function Base.promote_rule(::Type{GaussianWavePacket1D{Tλ, Tz, Tq, Tp}}, ::Type{TH}) where{Tλ, Tz, Tq, Tp, TH<:HermiteWavePacket1D}
    return promote_type(HermiteWavePacket1D{1, Tλ, Tz, Tq, Tp}, TH)
end

# 
function Base.promote_rule(::Type{<:HermiteFct1D}, ::Type{HermiteWavePacket1D})
    return HermiteWavePacket1D
end
function Base.promote_rule(::Type{HermiteFct1D{N, TΛ, Ta, Tq}}, ::Type{TH}) where{N, TΛ, Ta, Tq, TH<:HermiteWavePacket1D}
    return promote_type(HermiteWavePacket1D{N, TΛ, Ta, Tq, Tq}, TH)
end


#=
    BASIC OPERATIONS
=#

# Returns a null hermite wave packet
@inline function zero(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(zero(SVector{N, TΛ}), one(Tz), zero(Tq), zero(Tp))
end
@inline function zero(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return zero(HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})
end

# Creates a copy of a hermite wave packet
@inline function copy(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(H.Λ, H.z, H.q, H.p)
end

#
function core_type(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return promote_type(TΛ, Tz, Tq, Tp)
end
function core_type(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return core_type(HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})
end

# 
function fitting_float(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return fitting_float(promote_type(TΛ, Tz, Tq, Tp))
end
function fitting_float(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return fitting_float(HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})
end

# Returns the complex conjugate of a hermite wave packet
@inline function conj(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(conj.(H.Λ), conj(H.z), H.q, -H.p)
end

# Evaluates a hermite wave packet at x
function (H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})(x::Number) where{N, TΛ, Tz, Tq, Tp}
    Ha = HermiteFct1D(H.Λ, real(H.z), H.q)
    e = cis(-imag(H.z) * (x - H.q)^2 / 2) * cis(x * H.p)
    return Ha(x) * e
end

# Evaluates a hermite function at all the points in x
function evaluate(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}, x::SVector{M, Tx}) where{N, TΛ, Tz, Tq, Tp, M, Tx<:Number}
    Ha = HermiteFct1D(H.Λ, real(H.z), H.q)
    e = @. cis(-imag(H.z) * (x - H.q)^2 / 2) * cis(x * H.p)
    return evaluate(Ha, x) .* e
end

#=
    TRANSFORMATIONS
=#


# Computes the product of a scalar and a gaussian
@inline function (*)(w::Number, H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(w .* H.Λ, H.z, H.q, H.p)
end

# Computes the product of two hermite wave packets
function (*)(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    N = max(N1 + N2 - 1, 0)
    z1, q1, p1 = H1.z, H1.q, H1.p
    z2, q2, p2 = H2.z, H2.q, H2.p
    z, q, p = complex_gaussian_product_arg(z1, q1, p1, z2, q2, p2)
    a, b = reim(z)
    x, _ = hermite_quadrature(a, q, Val(N))
    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = SVector{N}(Φ1[j] * Φ2[j] * cis(b * (x[j] - q)^2 / 2) * cis(- p * x[j]) for j in 1:N)
    Λ = hermite_discrete_transform(Φ, a, q, Val(N))
    return HermiteWavePacket1D(Λ, z, q, p)
end

#=
    Computes the product of a hermite wave packet with a polynomial
        P(x) = ∑ₖ P[k](x-q)^k
=#
function polynomial_product(q::Tq1, P::SVector{N1, TΛ1}, H::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{Tq1<:Real, N1, TΛ1, N2, TΛ2, Tz2, Tq2, Tp2}
    Ha = HermiteFct1D(H.Λ, real(H.z), H.q)
    PHa = polynomial_product(q, P, Ha)
    return HermiteWavePacket1D(PHa.Λ, H.z, H.q, H.p)
end

# Multiplies a hermite wave packet by exp(-ib/2 * (x - q)^2) * exp(ipx)
@inline function unitary_product(b::Real, q::Real, p::Real, H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    α = cis(b / 2 * (H.q + q) * (H.q - q))
    return HermiteWavePacket1D(α .* H.Λ, H.z + complex(0, b), H.q, H.p + p - b * (H.q - q))
end
# Multiplies a hermite wave packet by exp(-ib/2 * x^2)
@inline function unitary_product(b::Real, H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    α = cis(b / 2 * H.q^2)
    return HermiteWavePacket1D(α .* H.Λ, H.z + complex(0, b), H.q, H.p - b * H.q)
end

# Computes the integral of a hermite wave packet
function integral(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    Hf = fourier(H)
    return Hf(zero(fitting_float(H)))
end

#=
    Computes the Fourier transform of a hermite function
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
# Real variance
function fourier(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Real, Tq, Tp}
    T = fitting_float(H)
    a, q, p = H.z, H.q, H.p
    af, qf, pf = complex_gaussian_fourier_arg(a, q, p)
    root4 = @SVector [1, -1im, -1, 1im]
    Λf = T(sqrt(2π)) * cis(p*q) .* SVector{N}(root4[n % 4 + 1] * H.Λ[n+1] for n in 0:N-1)
    return HermiteWavePacket1D(Λf, af, qf, pf)
end
# Complex variance
function fourier(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Complex, Tq, Tp}
    T = fitting_float(H)

    z, q, p = H.z, H.q, H.p
    a = real(z)

    # Fourier transform of the first hermite function
    Gf = fourier(GaussianWavePacket1D(a^T(1/4), z, q, p))
    zf, qf, pf = Gf.z, Gf.q, Gf.p
    af = real(zf)

    # 
    λ0 = Gf.λ * af^T(-1/4)
    α = -im * conj(z) / abs(z)
    Λf = λ0 .* SVector{N}(α^n * H.Λ[n+1] for n in 0:N-1)

    return HermiteWavePacket1D(Λf, zf, qf, pf)
end

#=
    Computes the inverse Fourier transform of a hermite function
    The Fourier transform is defined as
        ITF(ψ)(ξ) = (2π)⁻¹ ∫dx e^(ixξ) ψ(x)
=#
# Real variance
function inv_fourier(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Real, Tq, Tp}
    T = fitting_float(H)
    a, q, p = H.z, H.q, H.p
    af, qf, pf = complex_gaussian_inv_fourier_arg(a, q, p)
    root4 = @SVector [1, 1im, -1, -1im]
    Λf = T((2π)^(-1/2)) * cis(p*q) .* SVector{N}(root4[n % 4 + 1] * H.Λ[n+1] for n in 0:N-1)
    return HermiteWavePacket1D(Λf, af, qf, pf)
end
# Complex variance
function inv_fourier(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Complex, Tq, Tp}
    T = fitting_float(H)

    z, q, p = H.z, H.q, H.p
    a = real(z)

    # Fourier transform of the first hermite function
    Gf = inv_fourier(GaussianWavePacket1D(a^T(1/4), z, q, p))
    zf, qf, pf = Gf.z, Gf.q, Gf.p
    af = real(zf)

    # 
    λ0 = Gf.λ * af^T(-1/4)
    α = im * conj(z) / abs(z)
    Λf = λ0 .* SVector{N}(α^n * H.Λ[n+1] for n in 0:N-1)

    return HermiteWavePacket1D(Λf, zf, qf, pf)
end

# Computes the convolution product of two hermite wave packets
@inline function convolution(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return inv_fourier(fourier(H1) * fourier(H2))
end

# Computes the L² product of two hermite wave packets
@inline function dot_L2(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return integral(conj(H1) * H2)
end
@inline function dot_L2(G1::GaussianWavePacket1D{Tλ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{Tλ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return dot_L2(HermiteWavePacket1D(G1), H2)
end
@inline function dot_L2(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, G2::GaussianWavePacket1D{Tλ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, Tλ2, Tz2, Tq2, Tp2}
    return dot_L2(H1, HermiteWavePacket1D(G2))
end

# Computes the square L² norm of a hermite wave packet
@inline function norm2_L2(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return real(dot(H.Λ, H.Λ))
end
# Computes the L² norm of a hermite wave packet
@inline function norm_L2(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return sqrt(norm2_L2(H))
end