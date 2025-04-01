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
struct HermiteWavePacket{D, N<:Tuple, TΛ<:Number, Tz<:Number, Tq<:Real, Tp<:Real, L} <: AbstractWavePacket{D}
    Λ::SArray{N, TΛ, D, L}
    z::SVector{D, Tz}
    q::SVector{D, Tq}
    p::SVector{D, Tp}
end

#=
    CONVERSIONS
=#

# function Base.convert(::Type{HermiteFct{D, N, TΛ, Ta, Tq}}, G::Gaussian{D}) where{N, TΛ, Ta, Tq, D}
#     T = fitting_float(G)
#     λ = convert(TΛ, T(π^(D/4)) * G.λ * prod(G.a)^T(-1/4))
#     Λ = SArray{N}(ifelse(all(k -> k==1, n), λ, zero(TΛ)) for n in Iterators.product((1:Nj for Nj in N.parameters)...))
#     return HermiteFct(Λ, Ta.(G.a), Tq.(G.q))
# end

# @generated function HermiteFct(G::Gaussian{D, TΛ, Ta, Tq}) where{D, TΛ, Ta, Tq}
#     N = Tuple{ntuple(_ -> 1, D)...}
#     return :( return convert(HermiteFct{$N, TΛ, Ta, Tq}, G) )
# end

function truncate_to_gaussian(H::HermiteWavePacket{D}) where D
    T = fitting_float(H)
    return GaussianWavePacket(T(π^(-D/4)) * prod(H.z)^T(1/4) * first(H.Λ), H.z, H.q, H.p)
end

#=
    PROMOTIONS
=#

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

#=
    BASIC OPERATIONS
=#

# Returns a null hermite function
function Base.zero(::Type{<:HermiteWavePacket{D, N, TΛ, Tz, Tq, Tp, L}}) where{D, N, TΛ, Tz, Tq, Tp, L}
    return HermiteWavePacket(zeros(SArray{N, TΛ}), ones(SVector{D, Tz}), zeros(SVector{D, Tq}), zeros(SVector{D, Tp}))
end

# Creates a copy of a gaussian
function Base.copy(H::HermiteWavePacket)
    return HermiteWavePacket(H.Λ, H.z, H.q, H.p)
end

#
function core_type(::Type{<:HermiteWavePacket{D, N, TΛ, Tz, Tq, Tp, L}}) where{D, N, TΛ, Tz, Tq, Tp, L}
    return promote_type(TΛ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a hermite function
function Base.conj(H::HermiteWavePacket)
    return HermiteWavePacket(conj.(H.Λ), conj.(H.z), H.q, .- H.p)
end

# Computes the product of a scalar and a hermite function
function Base.:-(H::HermiteWavePacket)
    return HermiteWavePacket(.- H.Λ, H.z, H.q, H.p)
end

# Computes the product of a scalar and a hermite function
function Base.:*(w::Number, H::HermiteWavePacket)
    return HermiteWavePacket(w .* H.Λ, H.z, H.q, H.p)
end

# Computes the division of a hermite function by a scalar
function Base.:/(H::HermiteWavePacket, w::Number)
    return HermiteWavePacket(H.Λ ./ w, H.z, H.q, H.p)
end

# Computes the square L² norm of a hermite function
function norm2_L2(H::HermiteWavePacket)
    return sum(abs2, H.Λ; init=zero(real(eltype(H.Λ))))
end