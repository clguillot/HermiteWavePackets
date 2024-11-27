import Base.*
import Base.copy
import Base.zero
import Base.conj

#=
    Reminder :
    -hermiteh(n, x) computes Hₙ(x)
     where the (Hₙ)ₙ are the Hermite polynomials orthognal with respect to the weight exp(-x²)
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
struct HermiteWavePacket1D{N, TΛ<:Number, Tz<:Number, Tq<:Real, Tp<:Real}
    Λ::SVector{N, TΛ}
    z::Tz
    q::Tq
    p::Tp
end

#=
    BASIC OPERATIONS
=#

# Returns a null hermite wave packet
@inline function zero(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(zero(SVector{N, TΛ}), one(Tz), zero(Tq), zero(Tp))
end

# Creates a copy of a hermite wave packet
@inline function copy(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(H.Λ, H.z, H.q, H.p)
end

# 
@generated function fitting_float(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    T = fitting_float(promote_type(TΛ, Tz, Tq, Tp))
    return :( $T )
end
@generated function fitting_float(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    T = fitting_float(HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})
    return :( $T )
end

# Returns the complex conjugate of a hermite wave packet
@inline function conj(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(conj.(H.Λ), conj(H.z), H.q, -H.p)
end

# Evaluates a hermite wave packet at x
function (H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp})(x::Number) where{N, TΛ, Tz, Tq, Tp}
    T = promote_type(fitting_float(H), fitting_float(x))

    a = real(H.z)
    u = T(π^(-1/4)) * a^T(1/4) * exp(-H.z * (x - H.q)^2 / 2) * cis(x * H.p)

    if N > 0
        val = H.Λ[1] * u
        if N > 1
            b = (2*a)^T(1/2)

            v = u
            u = b * (x - H.q) * u
            val += H.Λ[2] * u

            for k=3:N
                w = u
                u = (b * (x - H.q) * u - sqrt(T(k-2)) * v) / sqrt(T(k-1))
                v = w
                val += H.Λ[k] * u
            end
        end
        return val
    else
        return zero(TΛ) * u
    end
end

# Evaluates a hermite function at all the points in x
function evaluate(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}, x::SVector{M, Tx}) where{N, TΛ, Tz, Tq, Tp, M, Tx<:Number}
    T = promote_type(fitting_float(H), fitting_float(x))

    a = real(H.z)
    u = @. T(π^(-1/4)) * a^T(1/4) * exp(-H.z * (x - H.q)^2 / 2) * cis(x * H.p)

    if N > 0
        val = @. H.Λ[1] * u
        if N > 1
            b = sqrt(2*a)
            v = u
            u = @. b * (x - H.q) * u
            val = @. val + H.Λ[2] * u

            for k=3:N
                w = u
                u = @. (b * (x - H.q) * u - sqrt(T(k-2)) * v) / sqrt(T(k-1))
                v = w
                val = @. val + H.Λ[k] * u
            end
        end
        return val
    else
        return zero(TΛ) .* u
    end
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
    x, M = hermite_discrete_transform(a, q, Val(N))
    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = SVector{N}(Φ1[j] * Φ2[j] * cis(b * (x[j] - q)^2 / 2) * cis(- p * x[j]) for j in 1:N)
    Λ = M * Φ
    return HermiteWavePacket1D(Λ, z, q, p)
end

# Computes the integral of a hermite wave packet
# Real variance
function integral(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Real, Tq, Tp}
    T = fitting_float(H)
    a, q, p = H.z, H.q, H.p
    
    u = T(π^(-1/4)) * a^T(1/4) * integral(GaussianWavePacket1D(one(T), a, q, p))
    
    if N > 0
        val = H.Λ[1] * u
        if N > 1
            c = sqrt(2 / a)
            v = u
            u = 1im * c * p * u
            val += H.Λ[2] * u
            for k=3:N
                w = u
                u = (1im * c * p * u + sqrt(T(k-2)) * v) / sqrt(T(k-1))
                v = w
                val += H.Λ[k] * u
            end
        end
        return val
    else
        return zero(TΛ) * u
    end
end
# Complex variance
function integral(H::HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}) where{N, TΛ, Tz<:Complex, Tq, Tp}
    T = fitting_float(H)
    z, q, p = H.z, H.q, H.p
    a = real(z)
    
    u = integral(GaussianWavePacket1D(T(π^(-1/4)) * a^T(1/4), z, q, p))
    
    if N > 0
        val = H.Λ[1] * u
        if N > 1
            b = sqrt(2*a)
            v = u
            u = 1im * b * p / z * u
            val += H.Λ[2] * u
            for k=3:N
                w = u
                u = (1im * b * p * u + conj(z) * sqrt(T(k-2)) * v) / (z * sqrt(T(k-1)))
                v = w
                val += H.Λ[k] * u
            end
        end
        return val
    else
        return zero(TΛ) * u
    end
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

# Computes the convolution product of two gaussians
function convolution(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return inv_fourier(fourier(H1) * fourier(H2))
end

# Computes the L² product of two gaussians
@inline function dot_L2(H1::HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return integral(conj(H1) * H2)
end