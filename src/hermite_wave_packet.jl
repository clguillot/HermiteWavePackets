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

    function HermiteWavePacket(Λ::SArray{N, TΛ, D, L}, z::SVector{D, Tz},
                    q::SVector{D, Tq} = zeros(SVector{D, NullNumber}),
                    p::SVector{D, Tp} = zeros(SVector{D, NullNumber})) where{D, N, TΛ<:Number, Tz<:Number, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}, L}
        return new{D, N, TΛ, Tz, Tq, Tp, L}(Λ, z, q, p)
    end

    function HermiteWavePacket(Λ::SVector{N, TΛ}, z::Tz, q::Tq=NullNumber(), p::Tp=NullNumber()) where{N, TΛ<:Number, Tz<:Number, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}}
        return HermiteWavePacket(Λ, SVector(z), SVector(q), SVector(p))
    end
end

@generated function HermiteWavePacket(G::GaussianWavePacket{D, Tλ, Tz, Tq, Tp}) where{D, Tλ, Tz, Tq, Tp}
    N = Tuple{fill(1, D)...}
    return :( return convert(HermiteWavePacket{D, $N, Tλ, Tz, Tq, Tp}, G) )
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
const HermiteFct{D, N<:Tuple, TΛ<:Number, Tz<:Real, Tq<:Union{Real, NullNumber}, L} =
        HermiteWavePacket{D, N, TΛ, Tz, Tq, NullNumber, L}
function HermiteFct(Λ::SArray{N, <:Number, D}, z::SVector{D, <:Real},
                q::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber})) where{D, N}
    return HermiteWavePacket(Λ, z, q, zeros(SVector{D, NullNumber}))
end
function HermiteFct(Λ::SVector{N, <:Number}, z::Real,
                q::Union{Real, NullNumber}=NullNumber()) where N
    return HermiteFct(Λ, SVector(z), SVector(q))
end

function HermiteFct(G::Gaussian)
    return HermiteWavePacket(G)
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{<:HermiteWavePacket{D, N, TΛ, Tz, Tq, Tp}}, G::GaussianWavePacket{D}) where{D, N, TΛ, Tz, Tq, Tp}
    λ = convert(TΛ, G.λ * prod(invπ * real.(G.z))^Rational(-1, 4))
    Λ = SArray{N}(ifelse(all(k -> k==1, n), λ, zero(TΛ)) for n in Iterators.product((1:Nj for Nj in N.parameters)...))
    return HermiteWavePacket(Λ, convert.(Tz, G.z), convert.(Tq, G.q), convert.(Tp, G.p))
end

function truncate_to_gaussian(H::HermiteWavePacket{D}) where D
    return GaussianWavePacket(prod(invπ * real.(H.z))^Rational(1, 4) * first(H.Λ), H.z, H.q, H.p)
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:HermiteWavePacket}, ::Type{HermiteWavePacket})
    return HermiteWavePacket
end
function promote_rule(::Type{HermiteWavePacket{D, N, TΛ1, Tz1, Tq1, Tp1, L}},
                      ::Type{HermiteWavePacket{D, N, TΛ2, Tz2, Tq2, Tp2, L}}) where{D, N, TΛ1, Tz1, Tq1, Tp1, TΛ2, Tz2, Tq2, Tp2, L}
    return HermiteWavePacket{D, N, promote_type(TΛ1, TΛ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2), L}
end

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
function core_type(::Type{HermiteWavePacket{D, N, TΛ, Tz, Tq, Tp, L}}) where{D, N, TΛ, Tz, Tq, Tp, L}
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
    
    zs = zero(SVector{D, Bool})
    zt = tuple((nothing for _ in 1:D)...)
    Hq_broad_tuple = [fill(:(H.q[$k]), length(y)) for (y, k) in zip(x.parameters, eachindex(zs))]
    Hq_broad = [:( SVector{$(length(y))}($(h...)) ) for (y, h) in zip(x.parameters, Hq_broad_tuple)]
    Hp_broad_tuple = [fill(:(H.p[$k]), length(y)) for (y, k) in zip(x.parameters, eachindex(zs))]
    Hp_broad = [:( SVector{$(length(y))}($(h...)) ) for (y, h) in zip(x.parameters, Hp_broad_tuple)]
    expr_x = [:( x[$k] .- $(Hq_broad[k]) ) for k in eachindex(zt)]
    expr_α = [:( exp.(.-H.z[$k]./2 .* (x[$k] .- $(Hq_broad[k])).^2 .+ im .* x[$k] .* $(Hp_broad[k])) ) for k in eachindex(zt)]
    μ = SVector{D}((true for _ in 1:D)...)
    return :( (prod(invπ * real.(H.z))^Rational(1, 4)) .* clenshaw_hermite_transform_grid(H.Λ, (@. sqrt(2 * real(H.z))), tuple($(expr_x...)), $μ, tuple($(expr_α...))))
end

# Evaluates a hermite function at x using Clenshaw's algorithm
function (H::HermiteWavePacket{D})(x::AbstractVector{<:Union{Number, NullNumber}}) where D
    xs = SVector{D}(x)
    return first(evaluate_grid(H, tuple((SVector{1}(y) for y in xs)...)))
end
function (H::HermiteWavePacket{1})(x::Union{Number, NullNumber})
    return H(SVector(x))
end

#=
    TRANSFORMATIONS
=#

# Multiplies a gaussian wave packet by exp(-i∑ₖbₖ/2 * (xₖ - qₖ)^2) * exp(ipx)
function unitary_product(H::HermiteWavePacket{D}, b::SVector{D, <:Union{Real, NullNumber}},
                q::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber}),
                p::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber})) where D
    G = unitary_product(GaussianWavePacket(true, H.z, H.q, H.p), b, q, p)
    return HermiteWavePacket(G.λ .* H.Λ, G.z, G.q, G.p)
end

# Computes the product of two hermite functions
@generated function Base.:*(H1::HermiteWavePacket{D, N1, TΛ1, Tz1}, H2::HermiteWavePacket{D, N2, TΛ2, Tz2}) where{D, N1, TΛ1, Tz1<:Real, N2, TΛ2, Tz2<:Real}
    N = Tuple{(@. max(N1.parameters + N2.parameters - 1, 0))...}
    code =
        quote
            z, q, p = gaussian_product_arg(H1.z, H1.q, H1.p, H2.z, H2.q, H2.p)

            x = hermite_grid(z, q, $N)
            Φ1 = evaluate_grid(H1, x)
            Φ2 = evaluate_grid(H2, x)
            Λ = hermite_discrete_transform(Φ1 .* Φ2, z)

            return HermiteWavePacket(Λ, z, q, p)
        end
    return code
end
function Base.:*(H1::HermiteWavePacket{D, N1, TΛ1, Tz1}, H2::HermiteWavePacket{D, N2, TΛ2, Tz2}) where{D, N1, TΛ1, Tz1, N2, TΛ2, Tz2}
    z, q, p = gaussian_product_arg(H1.z, H1.q, H1.p, H2.z, H2.q, H2.p)
    Hr = HermiteFct(H1.Λ, real.(H1.z), H1.q) * HermiteFct(H2.Λ, real.(H2.z), H2.q)
    G1 = GaussianWavePacket(true, im * imagz.(H1.z), H1.q, H1.p)
    G2 = GaussianWavePacket(true, im * imagz.(H2.z), H2.q, H2.p)
    λ = G1(q) * G2(q) * cis(-dot(q, p))
    return HermiteWavePacket(λ * Hr.Λ, z, q, p)
end

#=
    Computes the product of a hermite function with a polynomial
        P(x) = ∑ₖ P[k](x-q)^k
=#
@generated function polynomial_product(H::HermiteFct{D, N}, P::SArray{NP, <:Number, D},
                        q::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber})) where{D, N, NP}
    N1 = Tuple{(max(n1+n2-1, 0) for (n1, n2) in zip(N.parameters, NP.parameters))...}
    λ = SVector{D}(ntuple(_ -> true, D)...)
    code =
        quote
            x = hermite_grid(H.z, H.q, $N1)
            Φ = evaluate_grid(H, x)
            Φ_P = horner_transform_grid(P, $λ, q, x)
            Λ = hermite_discrete_transform(Φ .* Φ_P, H.z)
            return HermiteFct(Λ, H.z, H.q)
        end
    return code
end
function polynomial_product(H::HermiteWavePacket{D, N}, P::SArray{NP, <:Number, D},
                q::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber})) where{D, N, NP}
    Hr = polynomial_product(HermiteFct(H.Λ, real.(H.z), H.q), P, q)
    return HermiteWavePacket(Hr.Λ, H.z, H.q, H.p)
end

# Computes the integral of a hermite function
function integral(H::HermiteFct{D}) where D
    z = zeros(SVector{D, NullNumber})
    x = tuple((SVector(NullNumber()) for _ in 1:D)...)
    μ = SVector{D}((-1 for _ in 1:D)...)
    α0 = tuple((SVector(true) for _ in 1:D)...)
    val = first(clenshaw_hermite_transform_grid(H.Λ, z, x, μ, α0))
    return prod(inv4π * H.z)^Rational(-1, 4) * val
end
function integral(H::HermiteWavePacket{D}) where D
    return fourier(H)(zeros(SVector{D, NullNumber}))
end

# Computes the Fourier transform of a hermite function with real variance
@generated function fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Real}
    T = fitting_float(H)
    M = SArray{N}((Complex{Int8}((-im)^sum(j)) for j in Iterators.product((0:n-1 for n in N.parameters)...))...)
    code =
        quote
            zf, qf, pf = gaussian_fourier_arg(H.z, H.q, H.p)
            return HermiteWavePacket(($T((2π)^($D/2)) * cis(dot(H.p, H.q))) .* $M .* H.Λ, zf, qf, pf)
        end
    return code
end
# Complex variance
@generated function fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz}
    zs = zeros(SVector{D, Bool})
    expr_d = [(:( α[$k]^$j ) for j in 0:n-1) for (n, k) in zip(N.parameters, eachindex(zs))]
    expr_D = [:( Diagonal(SVector{$n}($(d...))) ) for (n, d) in zip(N.parameters, expr_d)]
    code =
        quote
            Gf = fourier(GaussianWavePacket(prod(real.(H.z))^Rational(1, 4), H.z, H.q, H.p))            
            α = @. - im * conj(H.z) / abs(H.z)
            Λf = static_tensor_transform(H.Λ, tuple($(expr_D...)))
            return HermiteWavePacket((Gf.λ * prod(real.(Gf.z))^Rational(-1, 4)) .* Λf, Gf.z, Gf.q, Gf.p)
        end
    return code
end

# Computes the inverse Fourier transform of a hermite function with real variance
@generated function inv_fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz<:Real}
    T = fitting_float(H)
    M = SArray{N}((Complex{Int8}(im^sum(j)) for j in Iterators.product((0:n-1 for n in N.parameters)...))...)
    code =
        quote
            zf, qf, pf = gaussian_inv_fourier_arg(H.z, H.q, H.p)
            return HermiteWavePacket(($T((2π)^(-$D/2)) * cis(dot(H.p, H.q))) .* $M .* H.Λ, zf, qf, pf)
        end
    return code
end
# Complex variance
@generated function inv_fourier(H::HermiteWavePacket{D, N, TΛ, Tz}) where{D, N, TΛ, Tz}
    zs = zeros(SVector{D, Bool})
    expr_d = [(:( α[$k]^$j ) for j in 0:n-1) for (n, k) in zip(N.parameters, eachindex(zs))]
    expr_D = [:( Diagonal(SVector{$n}($(d...))) ) for (n, d) in zip(N.parameters, expr_d)]
    code =
        quote
            Gf = inv_fourier(GaussianWavePacket(prod(real.(H.z))^Rational(1, 4), H.z, H.q, H.p))            
            α = @. im * conj(H.z) / abs(H.z)
            Λf = static_tensor_transform(H.Λ, tuple($(expr_D...)))
            return HermiteWavePacket((Gf.λ * prod(real.(Gf.z))^Rational(-1, 4)) .* Λf, Gf.z, Gf.q, Gf.p)
        end
    return code
end

# Convolution of two Hermite functions
function convolution(H1::HermiteFct{D, N1, TΛ1}, H2::HermiteFct{D, N2, TΛ2}) where{D, N1, TΛ1<:Real, N2, TΛ2<:Real}
    H = inv_fourier(fourier(H1) * fourier(H2))
    return HermiteFct(real.(H.Λ), H.z, H.q)
end
function convolution(H1::HermiteWavePacket, H2::HermiteWavePacket)
    return inv_fourier(fourier(H1) * fourier(H2))
end

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
            z, q, _ = gaussian_product_arg(H1.z, H1.q, H1.p, H2.z, H2.q, H2.p)
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
function dot_L2(G1::GaussianWavePacket{D}, G2::HermiteWavePacket{D}) where D
    return dot_L2(HermiteWavePacket(G1), G2)
end
function dot_L2(G1::HermiteWavePacket{D}, G2::GaussianWavePacket{D}) where D
    return dot_L2(G1, HermiteWavePacket(G2))
end

# Computes the square L² norm of a hermite function
function norm2_L2(H::HermiteWavePacket)
    return sum(abs2, H.Λ; init=zero(real(eltype(H.Λ))))
end