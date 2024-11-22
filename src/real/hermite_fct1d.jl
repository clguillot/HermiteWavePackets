import Base.*
import Base.copy
import Base.zero

export StaticHermiteFct1D
export evaluate
export integral
export convolution
export dot_L2

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
        ̂ψₙ = √(2πa) * (-i)ⁿ * exp(-iqξ) * ψₙ(1/a, 0, ξ)
        ??? -Normalized ψₙ₊₁ = (√2*ψₙ' + √n) / √(n+1)
=#

#=
    Represents the function
        ∑ₙ Λ[n+1]*ψₙ(a, q, x) (n=0,...,N-1)
=#
struct StaticHermiteFct1D{N, TΛ<:Number, Ta<:Real, Tq<:Real}
    Λ::SVector{N, TΛ}
    a::Ta
    q::Tq
end

#=
    BASIC OPERATIONS
=#

# Returns a null hermite function
@inline function zero(::Type{StaticHermiteFct1D{N, TΛ, Ta, Tq}}) where{N, TΛ, Ta, Tq}
    return StaticHermiteFct1D(zeros(SVector{N, TΛ}), one(Ta), zero(Tq))
end

# Creates a copy of a gaussian
@inline function copy(H::StaticHermiteFct1D)
    return StaticHermiteFct1D(H.Λ, H.a, H.q)
end

# Evaluates a hermite function at x
function (H::StaticHermiteFct1D{N, TΛ, Ta, Tq})(x::Tx) where{N, TΛ, Ta, Tq, Tx<:Number}
    T = fitting_float(promote_type(TΛ, Ta, Tq, Tx))

    u = (H.a/T(π))^(1/4) * myexp(-H.a * (x - H.q)^2 / 2)
    b = sqrt(2*H.a)

    if N > 0
        val = H.Λ[1] * u
        if N > 1
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
        return zero(eltype(H.Λ)) * u
    end
end

# Evaluates a hermite function at all the points in x
function evaluate(H::StaticHermiteFct1D{N, TΛ, Ta, Tq}, x::SVector{M, Tx}) where{N, TΛ, Ta, Tq, M, Tx<:Number}
    T = fitting_float(promote_type(TΛ, Ta, Tq, Tx))

    u = @. (H.a/T(π))^(1/4) * myexp(-H.a * (x - H.q)^2 / 2)
    b = sqrt(2*H.a)

    if N > 0
        val = @. H.Λ[1] * u
        if N > 1
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
        return @. zero(eltype(H.Λ)) * u
    end
end

# Computes the product of a scalar and a hermite function
@inline function (*)(μ::Real, H::StaticHermiteFct1D)
    return StaticHermiteFct1D(μ .* H.Λ, H.a, H.q)
end

# Computes the product of two hermite functions
function (*)(H1::StaticHermiteFct1D, H2::StaticHermiteFct1D)
    N1, a1, q1 = length(H1.Λ), H1.a, H1.q
    N2, a2, q2 = length(H2.Λ), H2.a, H2.q
    a, q = gaussian_product_arg(a1, q1, a2, q2)
    N = N1 + N2 - 1

    x, M = hermite_discrete_transform(a, q, Val(N))
    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = Φ1 .* Φ2
    Λ = M * Φ

    return StaticHermiteFct1D(Λ, a, q)
end

# Computes the integral of a hermite function
function integral(H::StaticHermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    T = fitting_float(promote_type(TΛ, Ta, Tq))
    return H.a^(-1/4) * dot(hermite_primitive_integral(T, Val(N)), H.Λ)
end

# Computes the convolution product of two hermite functions
function convolution(H1::StaticHermiteFct1D{N1, TΛ1, Ta1, Tq1}, H2::StaticHermiteFct1D{N2, TΛ2, Ta2, Tq2}) where{N1, TΛ1, Ta1, Tq1, N2, TΛ2, Ta2, Tq2}
    T = fitting_float(promote_type(TΛ1, Ta1, Tq1, TΛ2, Ta2, Tq2))
    N = N1 + N2 - 1
    
    # We compute the convolution as the inverse Fourier transform of
    #  the product of the Fourier transforms
    Λf1 = SVector{N1}((-1im)^n * H1.Λ[n+1] for n=0:N1-1)
    af1 = inv(H1.a)
    Hf1 = StaticHermiteFct1D(Λf1, af1, zero(T))
    Λf2 = SVector{N2}((-1im)^n * H2.Λ[n+1] for n=0:N2-1)
    af2 = inv(H2.a)
    Hf2 = StaticHermiteFct1D(Λf2, af2, zero(T))
    Hf = Hf1 * Hf2

    a = inv(Hf.a)
    q = H1.q + H2.q
    Λ = T(sqrt(2π)) .* real.(SVector{N}((1im)^n * Hf.Λ[n+1] for n=0:N-1))
    return StaticHermiteFct1D(Λ, a, q)
end

# Computes the L² product of two gaussians
function dot_L2(H1::StaticHermiteFct1D, H2::StaticHermiteFct1D)
    return integral(H1 * H2)
end