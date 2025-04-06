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
struct HermiteWavePacket{D, N<:Tuple, TΛ<:Number, Tz<:Number, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}, L} <: AbstractWavePacket{D}
    Λ::SArray{N, TΛ, D, L}
    z::SVector{D, Tz}
    q::SVector{D, Tq}
    p::SVector{D, Tp}
end

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

# 
@generated function evaluate_grid(H::HermiteWavePacket{D}, x::Tuple) where D
    if length(x.parameters) != D
        throw(DimensionMismatch("Expecting the length of x to be $D, but got $(length(x.parameters)) instead"))
    end
    for y in x.parameters
        if !(y <: SVector)
            throw(ArgumentError("All elements of `x` must be `SVector`"))
        end
    end

    T = fitting_float(H, eltype.(x.parameters)...)
    
    zs = zero(SVector{D, Bool})
    zt = tuple((nothing for _ in 1:D)...)
    Hq_broad_tuple = [fill(:(H.q[$k]), length(y)) for (y, k) in zip(x.parameters, eachindex(zs))]
    Hq_broad = [:( SVector{$(length(y))}($(h...)) ) for (y, h) in zip(x.parameters, Hq_broad_tuple)]
    Hp_broad_tuple = [fill(:(H.p[$k]), length(y)) for (y, k) in zip(x.parameters, eachindex(zs))]
    Hp_broad = [:( SVector{$(length(y))}($(h...)) ) for (y, h) in zip(x.parameters, Hp_broad_tuple)]
    expr_x = [:( x[$k] .- $(Hq_broad[k]) ) for k in eachindex(zt)]
    expr_α = [:( exp.(.-H.z[$k]./2 .* (x[$k] .- $(Hq_broad[k])).^2 .+ im .* x[$k] .* $(Hp_broad[k])) ) for k in eachindex(zt)]
    μ = SVector{D}((true for _ in 1:D)...)
    return :( ($T(π^(-$D/4)) * prod(real.(H.z))^$T(1/4)) .* clenshaw_hermite_transform_grid(H.Λ, (@. sqrt(2 * real(H.z))), tuple($(expr_x...)), $μ, tuple($(expr_α...))))
end

# Evaluates a hermite function at x using Clenshaw's algorithm
function (H::HermiteWavePacket{D})(x::AbstractVector{<:Number}) where D
    xs = SVector{D}(x)
    return first(evaluate_grid(H, tuple((SVector{1}(y) for y in xs)...)))
end

#=
    TRANSFORMATIONS
=#

# Multiplies a gaussian wave packet by exp(-i∑ₖbₖ/2 * (xₖ - qₖ)^2) * exp(ipx)
function unitary_product(b::AbstractVector{<:Real}, q::AbstractVector{<:Real}, p::AbstractVector{<:Real}, H::HermiteWavePacket{D}) where D
    b = SVector{D}(b)
    q = SVector{D}(q)
    p = SVector{D}(p)
    u = @. b * (H.q + q) * (H.q - q)
    Λ_ = cis(sum(u) / 2) .* H.Λ
    z_ = @. H.z + complex(0, b)
    q_ = H.q
    p_ = @. H.p + p - b * (H.q - q)
    return HermiteWavePacket(Λ_, z_, q_, p_)
end
# Multiplies a gaussian wave packet by exp(-ib/2 * x^2)
function unitary_product(b::AbstractVector{<:Real}, H::HermiteWavePacket{D}) where D
    return unitary_product(b, zeros(SVector{D, NullNumber}), zeros(SVector{D, NullNumber}), H)
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
function Base.:*(H1::HermiteWavePacket{D}, H2::HermiteWavePacket{D}) where D
    z, q, p = complex_gaussian_product_arg(H1.z, H1.q, H1.p, H2.z, H2.q, H2.z)
    Hr = HermiteFct(H1.Λ, real.(H1.z), H1.q) * HermiteFct(H2.Λ, real.(H2.z), H2.q)
    return unitary_product(imag.(z), q, p, Hr)
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
function polynomial_product(q::SVector{D, <:Number}, P::SArray{N1, <:Number, D}, H::HermiteWavePacket{D, N2}) where{D, N1, N2}
    Hr = polynomial_product(q, P, HermiteFct(H.Λ, real.(H.z), H.q))
    return unitary_product(imag.(H.z), H.q, H.p, Hr)
end

# Computes the integral of a hermite function
function integral(H::HermiteFct{D}) where D
    T = fitting_float(H)
    z = zeros(SVector{D, NullNumber})
    x = tuple((SVector(NullNumber()) for _ in 1:D)...)
    μ = SVector{D}((-1 for _ in 1:D)...)
    α0 = tuple((SVector(true) for _ in 1:D)...)
    val = first(clenshaw_hermite_transform_grid(H.Λ, z, x, μ, α0))
    return prod(H.z)^T(-1/4) * T((4π)^(D/4)) * val
end

# Computes the Fourier transform of a hermite function with real variance
@generated function fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Real}
    T = fitting_float(H)
    M = SArray{N}((Complex{Int8}((-im)^sum(j)) for j in Iterators.product((0:n-1 for n in N.parameters)...))...)
    code =
        quote
            zf, qf, pf = complex_gaussian_fourier_arg(H.z, H.q, H.p)
            return HermiteWavePacket($T((2π)^($D/2)) .* $M .* H.Λ, zf, qf, pf)
        end
    return code
end
# # Complex variance
# @generated function fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Complex}
#     
# end

# Computes the inverse Fourier transform of a hermite function with real variance
@generated function inv_fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Real}
    T = fitting_float(H)
    M = SArray{N}((Complex{Int8}(im^sum(j)) for j in Iterators.product((0:n-1 for n in N.parameters)...))...)
    code =
        quote
            zf, qf, pf = complex_gaussian_inv_fourier_arg(H.z, H.q, H.p)
            return HermiteWavePacket(($T((2π)^(-$D/2)) * cis(dot(H.p, H.q))) .* $M .* H.Λ, zf, qf, pf)
        end
    return code
end
# # Complex variance
# @generated function fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Complex}
#     
# end

# Convolution of two Hermite functions
function convolution(H1::HermiteFct{D, N1, TΛ1}, H2::HermiteFct{D, N2, TΛ2}) where{D, N1, TΛ1<:Real, N2, TΛ2<:Real}
    H = inv_fourier(fourier(H1) * fourier(H2))
    return HermiteFct(real.(H.Λ), H.z, H.q)
end
function convolution(H1::HermiteWavePacket, H2::HermiteWavePacket)
    return inv_fourier(fourier(H1) * fourier(H2))
end

#=

=#

#=
    Computes the L² product of two hermite functions
        ∫dx conj(H1(x)) H2(x)
=#
@generated function dot_L2(H1::HermiteFct{D, N1}, H2::HermiteFct{D, N2}) where{D, N1, N2}
    N = Tuple{(cld(n1+n2-1, 2) for (n1, n2) in zip(N1.parameters, N2.parameters))...}
    zt = ntuple(_ -> nothing, D)
    expr_w = [:( reshape(w[$k], $(Size(1, n))) ) for (k, n) in zip(eachindex(zt), N.parameters)]
    code =
        quote
            z, q = gaussian_product_arg(H1.z, H1.q, H2.z, H2.q)
            x, w = hermite_quadrature(z ./ 2, q, $N)
            wc = tuple($(expr_w...))

            Φ1 = evaluate_grid(H1, x)
            Φ2 = evaluate_grid(H2, x)

            return first(static_tensor_transform(conj.(Φ1) .* Φ2, wc))
        end
    return code
end
function dot_L2(H1::HermiteWavePacket, H2::HermiteWavePacket)
    return integral(conj(H1) * H2)
end

# Computes the square L² norm of a hermite function
function norm2_L2(H::HermiteWavePacket)
    return sum(abs2, H.Λ; init=zero(real(eltype(H.Λ))))
end