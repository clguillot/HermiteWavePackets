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
        ∑ₙ Λ[n]*ψₙ(a, q, x) (n=multi index)
=#
const HermiteFct{D, N<:Tuple, TΛ<:Number, Tz<:Real, Tq<:Real, L} =
        HermiteWavePacket{D, N, TΛ, Tz, Tq, NullNumber, L}

function HermiteFct(Λ::SArray{N, <:Number, D}, z::SVector{D, <:Real}, q::SVector{D, <:Real}) where{D, N}
    return HermiteWavePacket(Λ, z, q, zeros(SVector{D, NullNumber}))
end

# 
@generated function evaluate_grid(H::HermiteFct{D}, x::Tuple) where D
    if length(x.parameters) != D
        throw(DimensionMismatch("Expecting the length of x to be $D, but got $(length(x.parameters)) instead"))
    end
    for y in x.parameters
        if !(y <: SVector && eltype(y) <: Number)
            throw(ArgumentError("All elements of `x` must be `SVector` with numeric element types"))
        end
    end

    T = fitting_float(H, eltype.(x.parameters)...)
    m = T(π^(-D/4))
    
    zs = zero(SVector{D, Bool})
    zt = tuple((false for _ in 1:D)...)
    expr_x = [:( x[$kx] .- H.q[$k] ) for (kx, k) in zip(eachindex(zt), eachindex(zs))]
    expr_α = [:( @. exp(-H.z[$k]/2 * (x[$kx] - H.q[$k])^2) ) for (kx, k) in zip(eachindex(zt), eachindex(zs))]
    μ = SVector{D}((true for _ in 1:D)...)
    return :( ($m * prod(H.z)^$T(1/4)) .* clenshaw_hermite_transform_grid(H.Λ, sqrt.(2 .* H.z), tuple($(expr_x...)), $μ, tuple($(expr_α...))))
end

# Evaluates a hermite function at x using Clenshaw's algorithm
function (H::HermiteFct{D})(x::AbstractVector{<:Number}) where D
    xs = SVector{D}(x)
    return first(evaluate_grid(H, tuple((SVector{1}(y) for y in xs)...)))
end

# Computes the product of two hermite functions
@generated function Base.:*(H1::HermiteFct{D, N1}, H2::HermiteFct{D, N2}) where{D, N1, N2}
    N = Tuple{(@. max(N1.parameters + N2.parameters - 1, 0))...}
    code =
        quote
            z, q = gaussian_product_arg(H1.z, H1.q, H2.z, H2.q)

            x = hermite_grid(z, q, $N)
            Φ1 = evaluate_grid(H1, x)
            Φ2 = evaluate_grid(H2, x)
            Λ = hermite_discrete_transform(Φ1 .* Φ2, z)

            return HermiteFct(Λ, z, q)
        end
    return code
end

#=
    Computes the product of a hermite function with a polynomial
        P(x) = ∑ₖ P[k](x-q)^k
=#
@generated function polynomial_product(q::SVector{D, <:Number}, P::SArray{N1, <:Number, D}, H::HermiteFct{D, N2}) where{D, N1, N2}
    N = Tuple{(max(n1+n2-1, 0) for (n1, n2) in zip(N1.parameters, N2.parameters))...}
    λ = SVector{D}(ntuple(_ -> true, D)...)
    code =
        quote
            x = hermite_grid(H.z, H.q, $N)
            Φ = evaluate_grid(H, x)
            Φ_P = horner_transform_grid(P, $λ, q, x)
            Λ = hermite_discrete_transform(Φ .* Φ_P, H.z)
            return HermiteFct(Λ, H.z, H.q)
        end
    return code
end

# Computes the integral of a hermite function
function integral(H::HermiteFct{D, N}) where{D, N}
    T = fitting_float(H)
    z = SVector((true for _ in 1:D)...)
    x = tuple((SVector(false) for _ in 1:D)...)
    μ = SVector{D}((-1 for _ in 1:D)...)
    α0 = tuple((SVector(true) for _ in 1:D)...)
    val = first(clenshaw_hermite_transform_grid(H.Λ, z, x, μ, α0))
    return prod(H.z)^T(-1/4) * T((4π)^(D/4)) * val
end

#=
    Computes the L² product of two hermite functions
        ∫dx conj(H1(x)) H2(x)
=#
# @generated function dot_L2(H1::HermiteFct{D, N1}, H2::HermiteFct{D, N2}) where{D, N1, N2}
#     N = Tuple{(cld(n1+n2-1, 2) for (n1, n2) in zip(N1.parameters, N2.parameters))...}
#     code =
#         quote
#             a, q = gaussian_product_arg(H1.a, H1.q, H2.a, H2.q)
#             x, w = hermite_quadrature(a ./ 2, q, $N)

#             Φ1 = evaluate_grid(H1, x)
#             Φ2 = evaluate_grid(H2, x)

#             return static_tensor_contraction(Φ1 .* Φ2, w...)
#         end
#     return code
# end