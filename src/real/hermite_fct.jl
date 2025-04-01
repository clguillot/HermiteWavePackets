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
struct HermiteFct{D, N<:Tuple, TΛ<:Number, Ta<:Real, Tq<:Real, L} <: AbstractWavePacket
    Λ::SArray{N, TΛ, D, L}
    a::SVector{D, Ta}
    q::SVector{D, Tq}

    function HermiteFct{D, N, TΛ, Ta, Tq, L}(Λ::SArray{N, TΛ, D, L}, a::SVector{D, Ta}, q::SVector{D, Tq}) where{D, N, TΛ, Ta, Tq, L}
        return new{D, N, TΛ, Ta, Tq, L}(Λ, a, q)
    end
    function HermiteFct(Λ::SArray{N, TΛ, D, L}, a::SVector{D, Ta}, q::SVector{D, Tq}) where{D, N, TΛ, Ta, Tq, L}
        return HermiteFct{D, N, TΛ, Ta, Tq, L}(Λ, a, q)
    end
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{HermiteFct{D, N, TΛ, Ta, Tq}}, G::Gaussian{D}) where{N, TΛ, Ta, Tq, D}
    T = fitting_float(G)
    λ = convert(TΛ, T(π^(D/4)) * G.λ * prod(G.a)^T(-1/4))
    Λ = SArray{N}(ifelse(all(k -> k==1, n), λ, zero(TΛ)) for n in Iterators.product((1:Nj for Nj in N.parameters)...))
    return HermiteFct(Λ, Ta.(G.a), Tq.(G.q))
end

@generated function HermiteFct(G::Gaussian{D, TΛ, Ta, Tq}) where{D, TΛ, Ta, Tq}
    N = Tuple{ntuple(_ -> 1, D)...}
    return :( return convert(HermiteFct{$N, TΛ, Ta, Tq}, G) )
end

function truncate_to_gaussian(H::HermiteFct)
    T = fitting_float(H)
    return Gaussian(T(π^(-1/4)) * prod(H.a)^T(1/4) * first(H.Λ), H.a, H.q)
end

# #=
#     PROMOTIONS
# =#

# # 
# function Base.promote_rule(::Type{<:HermiteFct1D}, ::Type{HermiteFct1D})
#     return HermiteFct1D
# end
# function Base.promote_rule(::Type{HermiteFct1D{N1, TΛ1, Ta1, Tq1}}, ::Type{HermiteFct1D{N2, TΛ2, Ta2, Tq2}}) where{N1, TΛ1, Ta1, Tq1, N2, TΛ2, Ta2, Tq2}
#     return HermiteFct1D{max(N1, N2), promote_type(TΛ1, TΛ2), promote_type(Ta1, Ta2), promote_type(Tq1, Tq2)}
# end

# # 
# function Base.promote_rule(::Type{<:Gaussian1D}, ::Type{HermiteFct1D})
#     return HermiteFct1D
# end
# function Base.promote_rule(::Type{Gaussian1D{Tλ, Ta, Tq}}, ::Type{TH}) where{Tλ, Ta, Tq, TH<:HermiteFct1D}
#     promote_type(HermiteFct1D{1, Tλ, Ta, Tq}, TH)
# end

# #=
#     BASIC OPERATIONS
# =#

# Returns a null hermite function
@inline function Base.zero(::Type{<:HermiteFct{D, N, TΛ, Ta, Tq, L}}) where{D, N, TΛ, Ta, Tq, L}
    return HermiteFct((zero(SArray{N, TΛ})), (SVector{D}(ones(Ta, D))), (SVector{D}(zeros(Tq, D))))
end

# Creates a copy of a gaussian
@inline function Base.copy(H::HermiteFct)
    return HermiteFct(H.Λ, H.a, H.q)
end

#
function core_type(::Type{<:HermiteFct{D, N, TΛ, Ta, Tq}}) where{D, N, TΛ, Ta, Tq}
    return promote_type(TΛ, Ta, Tq)
end

# Returns the complex conjugate of a hermite function
@inline function Base.conj(H::HermiteFct)
    return HermiteFct(conj.(H.Λ), H.a, H.q)
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
    expr_α = [:( @. exp(-H.a[$k]/2 * (x[$kx] - H.q[$k])^2) ) for (kx, k) in zip(eachindex(zt), eachindex(zs))]
    μ = SVector{D}((true for _ in 1:D)...)
    return :( ($m * prod(H.a)^$T(1/4)) .* clenshaw_hermite_transform_grid(H.Λ, sqrt.(2 .* H.a), tuple($(expr_x...)), $μ, tuple($(expr_α...))))
end

# Evaluates a hermite function at x using Clenshaw's algorithm
function (H::HermiteFct{D})(x::AbstractVector{<:Number}) where D
    xs = SVector{D}(x)
    return first(evaluate_grid(H, tuple((SVector{1}(y) for y in xs)...)))
end

# Computes the product of a scalar and a hermite function
@inline function Base.:-(H::HermiteFct)
    return HermiteFct(.- H.Λ, H.a, H.q)
end

# Computes the product of a scalar and a hermite function
@inline function Base.:*(w::Number, H::HermiteFct)
    return HermiteFct(w .* H.Λ, H.a, H.q)
end

# Computes the division of a hermite function by a scalar
@inline function Base.:/(H::HermiteFct, w::Number)
    return HermiteFct(H.Λ ./ w, H.a, H.q)
end

# Computes the product of two hermite functions
@generated function Base.:*(H1::HermiteFct{D, N1}, H2::HermiteFct{D, N2}) where{D, N1, N2}
    N = Tuple{(@. max(N1.parameters + N2.parameters - 1, 0))...}
    code =
        quote
            a, q = gaussian_product_arg(H1.a, H1.q, H2.a, H2.q)

            x = hermite_grid(a, q, $N)
            Φ1 = evaluate_grid(H1, x)
            Φ2 = evaluate_grid(H2, x)
            Λ = hermite_discrete_transform(Φ1 .* Φ2, a)

            return HermiteFct(Λ, a, q)
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
            x = hermite_grid(H.a, H.q, $N)
            Φ = evaluate_grid(H, x)
            Φ_P = horner_transform_grid(P, $λ, q, x)
            Λ = hermite_discrete_transform(Φ .* Φ_P, H.a)
            return HermiteFct(Λ, H.a, H.q)
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
    return prod(H.a)^T(-1/4) * T((4π)^(D/4)) * val
end

# # Computes the convolution product of two hermite functions
# function convolution(H1::HermiteFct1D{N1}, H2::HermiteFct1D{N2}) where{N1, N2}
#     T = promote_type(fitting_float(H1), fitting_float(H2))
#     N = max(N1 + N2 - 1, 0)
    
#     # We compute the convolution as the inverse Fourier transform of
#     #  the product of the Fourier transforms
#     Λf1 = SVector{N1}((-1im)^n * H1.Λ[n+1] for n=0:N1-1)
#     af1 = inv(H1.a)
#     Hf1 = HermiteFct1D(Λf1, af1, zero(T))
#     Λf2 = SVector{N2}((-1im)^n * H2.Λ[n+1] for n=0:N2-1)
#     af2 = inv(H2.a)
#     Hf2 = HermiteFct1D(Λf2, af2, zero(T))
#     Hf = Hf1 * Hf2

#     a = inv(Hf.a)
#     q = H1.q + H2.q
#     cond_real(z) = complex_truncation(promote_type(core_type(H1), core_type(H2)), z)
#     Λ = T(sqrt(2π)) .* SVector{N}(cond_real.((1im)^n * Hf.Λ[n+1]) for n=0:N-1)
#     return HermiteFct1D(Λ, a, q)
# end

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

# Computes the square L² norm of a hermite function
function norm2_L2(H::HermiteFct)
    return sum(abs2, H.Λ; init=zero(real(eltype(H.Λ))))
end