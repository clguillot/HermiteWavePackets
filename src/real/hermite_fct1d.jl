import Base.*
import Base.copy
import Base.zero

#=
    Reminder :
    -hermiteh(n, x) computes Hₙ(x)
     where the (Hₙ)ₙ are the Hermite polynomials orthognal with respect to the weight exp(-x²)
    -gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,m) such that the quadrature formula
     ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1

    We have the folowing properties (cf https://mathworld.wolfram.com/HermitePolynomial.html)
        ∫dx Hₘ(x)Hₙ(x)exp(-x²) = δₘₙ2ⁿn!√π
        Hₙ₊₁(x) = 2xHₙ(x) - 2nHₙ₋₁(x)
        Hₙ'(x) = 2n Hₙ₋₁(x)

    Now define for a > 0, and q a real number
        ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*(x - q)) * exp(-a(x - q)²/2)
    (we shall only write ψₙ(x) when the context is clear enough)
    then it follows from the previous properties that
        ∫ψₘψₙ = δₘₙ
        ψₙ₊₁ = (√(2a)*x*ψₙ - √n*ψₙ₋₁) / √(n+1)
        ??? -Normalized ψₙ₊₁ = (√2*ψₙ' + √n) / √(n+1)
=#

#=
    Represents the function
        ∑ₙ Λ[n+1]*ψₙ(a, q, x) (n=0,...,N-1)
=#
struct StaticHermiteFct{N, TΛ<:Number, Ta<:Real, Tq<:Real}
    Λ::SVector{N, TΛ}
    a::Tz
    q::Tq
end

#=
    BASIC OPERATIONS
=#

# Returns a null hermite function
@inline function zero(::Type{StaticHermiteFct{N, TΛ, Ta, Tq}}) where{N, TΛ, Ta, Tq}
    return StaticHermiteFct(zeros(SVector{N, TΛ}), one(Ta), zero(Tq))
end

# Creates a copy of a gaussian
@inline function copy(H::StaticHermiteFct)
    return StaticHermiteFct(H.Λ, H.a, H.q)
end

# Evaluates a hermite function at x
function (H::StaticHermiteFct)(x::Number)
    N = length(H.Λ)

    u = (H.a/π)^(1/4) * myexp(-H.a * (x - H.q)^2 / 2)
    b = mysqrt(2*a)

    if N > 0
        val = H.Λ[1] * u
        if N > 1
            v = u
            u = b * (x - q) * u
            val += H.Λ[2] * u

            for k=3:N
                w = u
                u = (b * (x - H.q) * u - sqrt(k-2) * v) / sqrt(k-1)
                v = w
                val += H.Λ[k] * u
            end
        end
        return val
    else
        return zero(eltype(H.Λ)) * u
    end
end

# Evaluates a hermite function at all the points in x
function evaluate(H::StaticHermiteFct, x::SVector{M, T}) where{M, T<:Number}
    N = length(H.Λ)

    u = @. (H.a/π)^(1/4) * myexp(-H.a * (x - H.q)^2 / 2)
    b = mysqrt(2*a)

    if N > 0
        val = @. H.Λ[1] * u
        if N > 1
            v = u
            u = @. b * (x - q) * u
            val = @. val + H.Λ[2] * u

            for k=3:N
                w = u
                u = @. (b * (x - H.q) * u - sqrt(k-2) * v) / sqrt(k-1)
                v = w
                val = @. val + H.Λ[k] * u
            end
        end
        return val
    else
        return @. zero(eltype(H.Λ)) * u
    end
end

# Computes the product of a scalar and a hermite function
@inline function (*)(μ::Real, H::StaticHermiteFct)
    return StaticHermiteFct(μ .* H.Λ, H.a, H.q)
end

# Computes the product of two hermite functions
function (*)(H1::StaticHermiteFct, H2::StaticHermiteFct)
    N1, a1, q1 = length(H1.Λ), H1.a, H1.q
    N2, a2, q2 = length(H2.Λ), H2.a, H2.q
    a, q = gaussian_product_arg(a1, q1, a2, q2)
    N = N1 + N2 - 1

    x, M = hermite_transform_matrix(a, q, Val(N))
    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = Φ1 .* Φ2
    Λ = M * Φ

    return StaticHermiteFct(Λ, a, q)
end

# Computes the integral of a hermite function
function integral(H::StaticHermiteFct)
    return dot(hermite_integral(H.a, H.q, Val(length(H.Λ))), H.Λ)
end

# Computes the convolution product of two hermite functions
function convolution(H1::StaticHermiteFct, H2::StaticHermiteFct)
    a1, q1 = H1.a, H1.q
    a2, q2 = H2.a, H2.q
    a, q = gaussian_convolution_arg(a1, q1, a2, q2)
end

# Computes the L² product of two gaussians
function dot_L2(H1::StaticHermiteFct, H2::StaticHermiteFct)
    return integral(G1 * G2)
end