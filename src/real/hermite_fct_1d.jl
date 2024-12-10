
#=
    Reminder :
    -gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,m) such that the quadrature formula
     ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1

    We have the folowing properties (cf https://mathworld.wolfram.com/HermitePolynomial.html)
        ∫dx Hₘ(x)Hₙ(x)exp(-x²) = δₘₙ2ⁿn!√π
        Hₙ₊₁(x) = 2xHₙ(x) - 2nHₙ₋₁(x)
        Hₙ'(x) = 2n Hₙ₋₁(x)

    Now define for a > 0, and q a real number
        ψₙ(a, q, x) = (a/π)^(1/4) / √(2ⁿn!) * Hₙ(√a*(x - q)) * exp(-a(x - q)²/2)
    (we shall only write ψₙ(x) when the context is clear enough)
    then it follows from the previous properties that
        ∫ψₘψₙ = δₘₙ
        ψₙ₊₁ = (√(2a)*x*ψₙ - √n*ψₙ₋₁) / √(n+1)
        ̂ψₙ = √(2π) * (-i)ⁿ * exp(-iqξ) * ψₙ(1/a, 0, ξ)
=#

#=
    Represents the function
        ∑ₙ Λ[n+1]*ψₙ(a, q, x) (n=0,...,N-1)
=#
struct HermiteFct1D{N, TΛ<:Number, Ta<:Real, Tq<:Real} <: AbstractWavePacket1D
    Λ::SVector{N, TΛ}
    a::Ta
    q::Tq
end

#=
    BASIC OPERATIONS
=#

#
function eltype(::Type{HermiteFct1D{N, TΛ, Ta, Tq}}) where{N, TΛ, Ta, Tq}
    return promote_type(TΛ, Ta, Tq)
end
function eltype(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return eltype(HermiteFct1D{N, TΛ, Ta, Tq})
end

# Returns a null hermite function
@inline function zero(::Type{HermiteFct1D{N, TΛ, Ta, Tq}}) where{N, TΛ, Ta, Tq}
    return HermiteFct1D(zeros(SVector{N, TΛ}), one(Ta), zero(Tq))
end

# Creates a copy of a gaussian
@inline function copy(H::HermiteFct1D)
    return HermiteFct1D(H.Λ, H.a, H.q)
end

# 
@generated function fitting_float(::Type{HermiteFct1D{N, TΛ, Ta, Tq}}) where{N, TΛ, Ta, Tq}
    Tf = fitting_float(promote_type(TΛ, Ta, Tq))
    return :( $Tf )
end
@generated function fitting_float(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    Tf = fitting_float(HermiteFct1D{N, TΛ, Ta, Tq})
    return :( $Tf )
end

# Returns the complex conjugate of a hermite function
@inline function conj(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return HermiteFct1D(conj.(H.Λ), H.a, H.q)
end

# Evaluates a hermite function at x using Clenshaw's algorithm
function (H::HermiteFct1D{N, TΛ, Ta, Tq})(x::Number) where{N, TΛ, Ta, Tq}
    T = promote_type(fitting_float(H), fitting_float(x))

    b = sqrt(2*H.a)
    ψ0 = T(π^(-1/4)) * (H.a)^T(1/4) * exp(-H.a * (x - H.q)^2 / 2)

    return clenshaw_hermite_eval(H.Λ, b, x - H.q, ψ0)
end

# Evaluates a hermite function at all the points in x
function evaluate(H::HermiteFct1D{N, TΛ, Ta, Tq}, x::SVector{M, Tx}) where{N, TΛ, Ta, Tq, M, Tx<:Number}
    T = promote_type(fitting_float(H), fitting_float(x))

    b = sqrt(2*H.a)
    ψ0 = @. T(π^(-1/4)) * (H.a)^T(1/4) * exp(-H.a * (x - H.q)^2 / 2)

    return clenshaw_hermite_eval(H.Λ, b, x .- H.q, ψ0)
end

# Computes the product of a scalar and a hermite function
@inline function (*)(w::Number, H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return HermiteFct1D(w .* H.Λ, H.a, H.q)
end

# Computes the product of two hermite functions
function (*)(H1::HermiteFct1D{N1, TΛ1, Ta1, Tq1}, H2::HermiteFct1D{N2, TΛ2, Ta2, Tq2}) where{N1, TΛ1, Ta1, Tq1, N2, TΛ2, Ta2, Tq2}
    a1, q1 = H1.a, H1.q
    a2, q2 = H2.a, H2.q
    a, q = gaussian_product_arg(a1, q1, a2, q2)
    N = N1 + N2 - 1

    x, M = hermite_discrete_transform(a, q, Val(N))
    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = Φ1 .* Φ2
    Λ = M * Φ

    return HermiteFct1D(Λ, a, q)
end

# Computes the integral of a hermite function
function integral(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    T = fitting_float(H)

    m = 2 * fld(N-1, 2) + 1
    val = zero(promote_type(TΛ, T))
    if N > 0
        val = H.Λ[m]
        for k=m-2:-2:1
            val = H.Λ[k] + sqrt(T(k) / T(k+1)) * val
        end
    end

    return H.a^T(-1/4) * T(sqrt(2) * π^(1/4)) * val
end

# Computes the convolution product of two hermite functions
function convolution(H1::HermiteFct1D{N1, TΛ1, Ta1, Tq1}, H2::HermiteFct1D{N2, TΛ2, Ta2, Tq2}) where{N1, TΛ1, Ta1, Tq1, N2, TΛ2, Ta2, Tq2}
    T = promote_type(fitting_float(H1), fitting_float(H2))
    N = max(N1 + N2 - 1, 0)
    
    # We compute the convolution as the inverse Fourier transform of
    #  the product of the Fourier transforms
    Λf1 = SVector{N1}((-1im)^n * H1.Λ[n+1] for n=0:N1-1)
    af1 = inv(H1.a)
    Hf1 = HermiteFct1D(Λf1, af1, zero(T))
    Λf2 = SVector{N2}((-1im)^n * H2.Λ[n+1] for n=0:N2-1)
    af2 = inv(H2.a)
    Hf2 = HermiteFct1D(Λf2, af2, zero(T))
    Hf = Hf1 * Hf2

    a = inv(Hf.a)
    q = H1.q + H2.q
    cond_real(z) = complex_truncation(promote_type(TΛ1, TΛ2), z)
    Λ = T(sqrt(2π)) .* SVector{N}(cond_real((1im)^n * Hf.Λ[n+1]) for n=0:N-1)
    return HermiteFct1D(Λ, a, q)
end

#=
    Computes the L² product of two hermite functions
        ∫dx conj(H1(x)) H2(x)
=#
function dot_L2(H1::HermiteFct1D{N1, TΛ1, Ta1, Tq1}, H2::HermiteFct1D{N2, TΛ2, Ta2, Tq2}) where{N1, TΛ1, Ta1, Tq1, N2, TΛ2, Ta2, Tq2}
    N = max(N1 + N2 - 1, 0)
    a, q = gaussian_product_arg(H1.a, H1.q, H2.a, H2.q)

    m = cld(N, 2)
    x, w = hermite_quadrature(a/2, q, Val(m))

    Φ1 = evaluate(H1, x)
    Φ2 = evaluate(H2, x)
    Φ = @. conj(Φ1) * Φ2

    return dot(w, Φ)
end

# Computes the square L² norm of a hermite function
@inline function norm2_L2(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return real(dot(H.Λ, H.Λ))
end
# Computes the L² norm of a hermite function
@inline function norm_L2(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
    return sqrt(norm2_L2(H))
end