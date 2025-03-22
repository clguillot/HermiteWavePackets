
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
    f(::Size{s}) where s = s
    S = f(Size(N))
    λ = convert(TΛ, T(π^(D/4)) * G.λ * prod(G.a)^T(-1/4))
    Λ = SArray{N}(ifelse(all(k -> k==1, n), λ, zero(TΛ)) for n in Iterators.product((1:Nj for Nj in S)...))
    return HermiteFct(Λ, Ta.(G.a), Tq.(G.q))
end

function HermiteFct1D(G::Gaussian{D, TΛ, Ta, Tq}) where{D, TΛ, Ta, Tq}
    return convert(HermiteFct{Tuple{}, TΛ, Ta, Tq}, G)
end

# function truncate_to_gaussian(H::HermiteFct1D)
#     T = fitting_float(H)
#     return Gaussian1D((H.a / π)^T(1/4) * first(H.Λ), H.a, H.q)
# end

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
@inline function Base.zero(::Type{HermiteFct{D, N, TΛ, Ta, Tq}}) where{D, N, TΛ, Ta, Tq}
    return HermiteFct((zeros(SArray{N, TΛ})), (@SVector ones(Ta, D)), (@SVector zeros(Tq, D)))
end

# Creates a copy of a gaussian
@inline function Base.copy(H::HermiteFct)
    return HermiteFct(H.Λ, H.a, H.q)
end

#
function core_type(::Type{HermiteFct{D, N, TΛ, Ta, Tq}}) where{D, N, TΛ, Ta, Tq}
    return promote_type(TΛ, Ta, Tq)
end

# Returns the complex conjugate of a hermite function
@inline function Base.conj(H::HermiteFct)
    return HermiteFct(conj.(H.Λ), H.a, H.q)
end

# # Evaluates a hermite function at x using Clenshaw's algorithm
# function (H::HermiteFct1D)(x::Number)
#     T = promote_type(fitting_float(H), fitting_float(x))

#     b = sqrt(2*H.a)
#     ψ0 = T(π^(-1/4)) * (H.a)^T(1/4) * exp(-H.a * (x - H.q)^2 / 2)

#     return clenshaw_hermite_eval(H.Λ, b, x - H.q, ψ0)
# end

# # Evaluates a hermite function at all the points in x
# function evaluate(H::HermiteFct1D, x::SVector{M, <:Number}) where M
#     T = promote_type(fitting_float(H), fitting_float(x))

#     b = sqrt(2*H.a)
#     ψ0 = @. T(π^(-1/4)) * (H.a)^T(1/4) * exp(-H.a * (x - H.q)^2 / 2)

#     return clenshaw_hermite_eval(H.Λ, b, x .- H.q, ψ0)
# end

# # Computes the product of a scalar and a hermite function
# @inline function Base.:-(H::HermiteFct1D)
#     return HermiteFct1D(-H.Λ, H.a, H.q)
# end

# # Computes the product of a scalar and a hermite function
# @inline function Base.:*(w::Number, H::HermiteFct1D)
#     return HermiteFct1D(w .* H.Λ, H.a, H.q)
# end

# # Computes the division of a hermite function by a scalar
# @inline function Base.:/(H::HermiteFct1D, w::Number)
#     return HermiteFct1D(H.Λ ./ w, H.a, H.q)
# end

# # Computes the product of two hermite functions
# function Base.:*(H1::HermiteFct1D{N1}, H2::HermiteFct1D{N2}) where{N1, N2}
#     a1, q1 = H1.a, H1.q
#     a2, q2 = H2.a, H2.q
#     a, q = gaussian_product_arg(a1, q1, a2, q2)
#     N = N1 + N2 - 1

#     x, _ = hermite_quadrature(a, q, Val(N))
#     Φ1 = evaluate(H1, x)
#     Φ2 = evaluate(H2, x)
#     Φ = Φ1 .* Φ2
    
#     Λ = hermite_discrete_transform(Φ, a, q, Val(N))

#     return HermiteFct1D(Λ, a, q)
# end

# #=
#     Computes the product of a hermite function with a polynomial
#         P(x) = ∑ₖ P[k](x-q)^k
# =#
# function polynomial_product(q::Tq1, P::SVector{N1, TΛ1}, H::HermiteFct1D{N2}) where{Tq1<:Real, N1, TΛ1, N2}
#     N = max(N1+N2-1, 0)
#     x, _ = hermite_quadrature(H.a, H.q, Val(N))
#     T = promote_type(eltype(x), Tq1, TΛ1)
    
#     Φ = evaluate(H, x)
    
#     Φ_P = zero(SVector{N, T})
#     for k in N1:-1:1
#         Φ_P = @. (x - q) * Φ_P + P[k]
#     end

#     Λ = hermite_discrete_transform(Φ .* Φ_P, H.a, H.q, Val(N))

#     return HermiteFct1D(Λ, H.a, H.q)
# end

# # Computes the integral of a hermite function
# function integral(H::HermiteFct1D{N, TΛ, Ta, Tq}) where{N, TΛ, Ta, Tq}
#     T = fitting_float(H)

#     m = 2 * fld(N-1, 2) + 1
#     val = zero(promote_type(TΛ, T))
#     if N > 0
#         val = H.Λ[m]
#         for k=m-2:-2:1
#             val = H.Λ[k] + sqrt(T(k) / T(k+1)) * val
#         end
#     end

#     return H.a^T(-1/4) * T(sqrt(2) * π^(1/4)) * val
# end

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

# #=
#     Computes the L² product of two hermite functions
#         ∫dx conj(H1(x)) H2(x)
# =#
# function dot_L2(H1::HermiteFct1D{N1}, H2::HermiteFct1D{N2}) where{N1, N2}
#     N = max(N1 + N2 - 1, 0)
#     a, q = gaussian_product_arg(H1.a, H1.q, H2.a, H2.q)

#     m = cld(N, 2)
#     x, w = hermite_quadrature(a/2, q, Val(m))

#     Φ1 = evaluate(H1, x)
#     Φ2 = evaluate(H2, x)
#     Φ = @. conj(Φ1) * Φ2

#     return dot(w, Φ)
# end
# @inline function dot_L2(G1::Gaussian1D, H2::HermiteFct1D)
#     return dot_L2(HermiteFct1D(G1), H2)
# end
# @inline function dot_L2(H1::HermiteFct1D, G2::Gaussian1D)
#     return dot_L2(H1, HermiteFct1D(G2))
# end

# # Computes the square L² norm of a hermite function
# @inline function norm2_L2(H::HermiteFct1D)
#     return real(dot(H.Λ, H.Λ))
# end