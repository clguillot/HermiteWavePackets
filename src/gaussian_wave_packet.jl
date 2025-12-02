
#=
    Represents the complex gaussian function
        λ*exp(-z(x-q)⋅(x-q)/2)*exp(ip⋅x)
=#
struct GaussianWavePacket{D, Tλ<:Number, Tz<:Union{Number, NullNumber}, Cz<:Union{Symmetric{Tz, <:SMatrix{D, D, Tz}}, SDiagonal{D, Tz}}, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}} <: AbstractWavePacket{D}
    λ::Tλ
    z::Cz
    q::SVector{D, Tq}
    p::SVector{D, Tp}

    function GaussianWavePacket(λ::Tλ, z::Union{Symmetric{Tz, <:SMatrix{D, D, Tz}}, SDiagonal{D, Tz}},
                q::SVector{D, Tq}=zeros(SVector{D, NullNumber}),
                p::SVector{D, Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}}
        return new{D, Tλ, Tz, typeof(z), Tq, Tp}(λ, z, q, p)
    end
    function GaussianWavePacket{D}(λ::Tλ, z::Symmetric{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber}),
                p::AbstractVector{Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}}
        return GaussianWavePacket(λ, Symmetric(SMatrix{D, D}(z)), SVector{D}(q), SVector{D}(p))
    end
    function GaussianWavePacket{D}(λ::Tλ, z::Diagonal{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber}),
                p::AbstractVector{Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}}
        return GaussianWavePacket(λ, Diagonal(SVector{D}(z.diag)), SVector{D}(q), SVector{D}(p))
    end
    function GaussianWavePacket(λ::Tλ, z::Tz, q::Tq=NullNumber(), p::Tp=NullNumber()) where{Tλ<:Number, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}, Tp<:Union{Real, NullNumber}}
        return GaussianWavePacket(λ, Diagonal(SVector(z)), SVector(q), SVector(p))
    end
end

#=
    Represents the gaussian function
        λ*exp(-z(x-q)⋅(x-q)/2)
=#
const Gaussian{D, Tλ<:Number, Tz<:Union{Real, NullNumber}, Cz<:Union{Symmetric{Tz, <:SMatrix{D, D, Tz}}}, Tq<:Union{Real, NullNumber}} =
            GaussianWavePacket{D, Tλ, Tz, Cz, Tq, NullNumber}
function Gaussian(λ::Tλ, z::Union{Symmetric{Tz, <:SMatrix{D, D, Tz}}, SDiagonal{D, Tz}},
                q::SVector{D, Tq}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}}
    return GaussianWavePacket(λ, z, q)
end
function Gaussian{D}(λ::Number, z::AbstractMatrix{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber})) where{D, Tz<:Union{Number, NullNumber}, Tq<:Union{Real, NullNumber}}
    return GaussianWavePacket(λ, z, q)
end
function Gaussian(λ::Number, z::Union{Real, NullNumber},
                q::Union{Real, NullNumber} = NullNumber())
    return Gaussian(λ, Diagonal(SVector(z)), SVector(q))
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Cz<:Diagonal, Tq, Tp}
    return GaussianWavePacket(convert(Tλ, G.λ), Diagonal(convert.(Tz, diag(G.z))), convert.(Tq, G.q), convert.(Tp, G.p))
end
function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Cz<:Symmetric, Tq, Tp}
    return GaussianWavePacket(convert(Tλ, G.λ), Symmetric(convert.(Tz, G.z.data)), convert.(Tq, G.q), convert.(Tp, G.p))
end

function truncate_to_gaussian(G::GaussianWavePacket)
    return G
end

#=
    PROMOTIONS
=#

# 
function Base.promote_rule(::Type{<:GaussianWavePacket}, ::Type{GaussianWavePacket})
    return GaussianWavePacket
end
function Base.promote_rule(::Type{GaussianWavePacket{D, Tλ1, Tz1, Cz1, Tq1, Tp1}}, ::Type{GaussianWavePacket{D, Tλ2, Tz2, Cz2, Tq2, Tp2}}) where{D, Tλ1, Tz1, Cz1, Tq1, Tp1, Tλ2, Tz2, Cz2, Tq2, Tp2}
    return GaussianWavePacket{D, promote_type(Tλ1, Tλ2), promote_type(Tz1, Tz2), promote_type(Cz1, Cz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end


#=
    BASIC OPERATIONS
=#

# Returns a null gaussian
function Base.zero(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}) where{D, Tλ, Tz, Cz, Tq, Tp}
    return GaussianWavePacket(zero(Tλ), zero(Cz), zeros(SVector{D, Tq}), zeros(SVector{D, Tp}))
end

# Creates a copy of a gaussian
function Base.copy(G::GaussianWavePacket)
    return GaussianWavePacket(G.λ, G.z, G.q, G.p)
end

#
function core_type(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}) where{D, Tλ, Tz, Cz, Tq, Tp}
    return promote_type(Tλ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a gaussian
function Base.conj(G::GaussianWavePacket)
    return GaussianWavePacket(conj(G.λ), conj(G.z), G.q, -G.p)
end

# 
function Base.:-(G::GaussianWavePacket)
    return GaussianWavePacket(-G.λ, G.z, G.q, G.p)
end

# Computes the product of a scalar and a gaussian
function Base.:*(w::Number, G::GaussianWavePacket)
    return GaussianWavePacket(w * G.λ, G.z, G.q, G.p)
end

# Computes the product of a gaussian by a scalar
function Base.:/(G::GaussianWavePacket, w::Number)
    return GaussianWavePacket(G.λ / w, G.z, G.q, G.p)
end

# Evaluates a gaussian at x
function evaluate(G::GaussianWavePacket{D},x::AbstractVector{<:Union{Number, NullNumber}}) where D
    xs = SVector{D}(x)
    ys = xs - G.q
    return G.λ * exp(- sum(ys .* (G.z * ys)) / 2) * cis(dot(G.p, xs))
end
# Evaluates a gaussian at x along the dimensions contained in N
# Preserves the order of the variables
@generated function evaluate(G::GaussianWavePacket{D, Tλ, Tz, Cz}, x::AbstractVector{<:Union{Number, NullNumber}}, ::Type{N}) where{D, Tλ, Tz, Cz, N}
    if !all(n -> n ∈ eachindex(zeros(SVector{D, Bool})), N.parameters)
        throw(DimensionMismatch("Cannot integrate over a dimension which does not exist"))
    end
    
    D2 = length(N.parameters)
    I1 = SVector((n for n ∈ eachindex(zeros(SVector{D, Bool})) if n ∉ N.parameters)...)
    I2 = SVector(N.parameters...)

    if length(N.parameters) == 0
        return :( G )
    elseif length(N.parameters) == D
        return :( evaluate(G, x) )
    elseif Cz <: Diagonal
        code = quote
            x2 = SVector{$D2}(x)
            z1 = Diagonal(diag(G.z)[$I1])
            z2 = Diagonal(diag(G.z)[$I2])
            G2 = GaussianWavePacket(G.λ, z2, G.q[$I2], G.p[$I2])
            return GaussianWavePacket(evaluate(G2, x2), z1, G.q[$I1], G.p[$I1])
        end
        return code
    else
        code = quote
            x2 = SVector{$D2}(x)
            z1 = Symmetric(G.z[$I1, $I1])
            z2 = Symmetric(G.z[$I2, $I2])
            q1 = G.q[$I1]
            q2 = G.q[$I2]
            w = G.z[$I1, $I2]
            U = special_cholesky(real(z1))
            y2 = x2 - q2
            y1 = real(w) * y2
            y1_ = U \ (transpose(U) \ y1)
            p0 = _imagz(w) * y2 - _imagz(z1) * y1_
            G2 = GaussianWavePacket(exp(dot(y1_, z1, y1_) / 2), z2, q2, G.p[$I2])
            λ = G.λ * evaluate(G2, x2) * cis(dot(p0, q1))
            return GaussianWavePacket(λ, z1, q1 - y1_, G.p[$I1] - p0)
        end
        return code
    end
end
# Evaluates a gaussian at x
(G::GaussianWavePacket{D})(x::AbstractVector{<:Union{Number, NullNumber}}) where D = evaluate(G, x)

#=
    TRANSFORMATIONS
=#

# Multiplies a gaussian wave packet by exp(-ib/2(x-q)⋅(x-q)) * exp(ipx)
function unitary_product(G::GaussianWavePacket{D}, b::Union{Symmetric{<:Union{Real, NullNumber}, <:SMatrix{D, D}}, SDiagonal{D, <:Union{Real, NullNumber}}},
            q::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber}),
            p::SVector{D, <:Union{Real, NullNumber}} = zeros(SVector{D, NullNumber})) where D
    z_ = G.z + im * b
    q_ = G.q
    p_ = G.p + p + b * (q - G.q)
    λ_ = G.λ * cis(dot(G.q - q, b, G.q + q) / 2)
    return GaussianWavePacket(λ_, z_, q_, p_)
end

# Computes the product of two gaussians
function Base.:*(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    z = G1.z + G2.z
    U = special_cholesky(real(z))
    q = U \ (transpose(U) \ (real(G1.z) * G1.q + real(G2.z) * G2.q))
    p0 = (_imagz(G1.z) * G1.q + _imagz(G2.z) * G2.q) - (_imagz(G1.z) + _imagz(G2.z)) * q
    p = G1.p + G2.p + p0
    λ = G1(q) * G2(q) * cis(-dot(p, q))
    return GaussianWavePacket(λ, z, q, p)
end

# Computes |G(x)|^2
Base.abs(G::GaussianWavePacket) = Gaussian(abs(G.λ), real(G.z), G.q)
Base.abs2(G::GaussianWavePacket) = Gaussian(abs2(G.λ), 2*real(G.z), G.q)

# Computes the integral of a gaussian
function integral(G::GaussianWavePacket)
    U = special_cholesky(G.z)
    r = transpose(U) \ G.p
    return G.λ / prod(invsqrt2π * diag(U)) * cis(dot(G.p, G.q)) * exp(-sum(r.^2) / 2)
end
# Integrate over all variables with indices contained in N,
#   while preserving the order of the other variables
function integral(G::GaussianWavePacket{D}, ::Type{N}) where{D, N}
    return inv_fourier(evaluate(fourier(G), zeros(SVector{length(N.parameters), NullNumber}), N))
end

#=
    Computes the Fourier transform of a gaussian
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
function fourier(G::GaussianWavePacket{D}) where D
    U = special_cholesky(G.z)
    U_inv = _upper_tri(U \ Diagonal(ones(SVector{D, Bool})))
    z_tf = _symmetric(U_inv * transpose(U_inv))
    q_tf = G.p
    p_tf = -G.q
    λ_tf = G.λ / prod(invsqrt2π * diag(U)) * cis(dot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

#=
    Computes the inverse Fourier transform of a gaussian
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻ᴰ∫dξ e^(ixξ) ψ(ξ)
=#
function inv_fourier(G::GaussianWavePacket{D}) where D
    U = special_cholesky(G.z)
    U_inv = _upper_tri(U \ Diagonal(ones(SVector{D, Bool})))
    z_tf = _symmetric(U_inv * transpose(U_inv))
    q_tf = -G.p
    p_tf = G.q
    λ_tf = G.λ / prod(sqrt2π * diag(U)) * cis(dot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

# Computes the convolution product of two gaussians
function convolution(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    return inv_fourier(fourier(G1) * fourier(G2))
end

# Computes the L² product of two gaussian wave packets
dot_L2(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D = integral(conj(G1) * G2)
# Computes the square L² norm of a gaussian wave packet
norm2_L2(G::GaussianWavePacket) = integral(abs2(G))

# Computes ∫<∇G1,∇G2> (where <,> is the hermitian product in ℂᴰ)
@generated function dot_∇(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    block = Expr(:block)
    deriv = Expr(:tuple)
    dim_list = eachindex(zeros(SVector{D, Bool}))
    for j in dim_list
        Rj = tuple((k for k in dim_list if k != j)...)
        Dj_symb = Symbol(:D, j)
        block_j = quote
            Gj = inv_fourier(evaluate(G, zeros(SVector{D-1, NullNumber}), Tuple{$Rj...}))
            zj = first(Gj.z)
            qj = first(Gj.q)
            pj = first(Gj.p)
            $Dj_symb = sqrt2π * Gj.λ / sqrt(zj) * (inv(zj) + (im*pj / zj + qj)^2) * exp(-pj^2/(2*zj)) * cis(qj*pj)
        end
        push!(block.args, block_j)
        push!(deriv.args, Dj_symb)
    end
    code = quote
        G = fourier(inv_fourier(conj(G1)) * fourier(G2))
        $block
        return sum($deriv)
    end
    return code
end

# Computes ∫|∇G|^2
@generated function norm2_∇(G::GaussianWavePacket{D}) where D
    block = Expr(:block)
    deriv = Expr(:tuple)
    dim_list = eachindex(zeros(SVector{D, Bool}))
    for j in dim_list
        Rj = tuple((k for k in dim_list if k != j)...)
        Dj_symb = Symbol(:D, j)
        block_j = quote
            Gj = inv_fourier(evaluate(G, zeros(SVector{D-1, NullNumber}), Tuple{$Rj...}))
            aj = first(Gj.z)
            qj = first(Gj.q)
            $Dj_symb = sqrt2π * Gj.λ * aj^Rational(-3, 2) * (aj*qj^2 + 1)
        end
        push!(block.args, block_j)
        push!(deriv.args, Dj_symb)
    end
    code = quote
        G = fourier(abs2(fourier(G)))
        $block
        I = sum($deriv)
        for j in 1:D
            I *= inv2π
        end
        return I
    end
    return code
end

# Computes ∫dx 1/|x| G(x) where G is an isotropic 3d gaussian wave packet
function coulomb_integral(G::GaussianWavePacket{3, <:Number, <:Number, <:SDiagonal})
    z0 = first(diag(G.z))
    if z0 != G.z[2, 2] || z0 != G.z[3, 3]
        throw(ArgumentError("Coulomb integral can only be computed for isotropic wave packets"))
    end
    r = sqrt(sum(x -> x^2, G.q + im * G.p / z0))
    if r == 0.0
        return fourπ * G.λ * cis(dot(G.q, G.p)) * exp(-sum(abs2, G.p) / (2*z0)) / z0
    else
        return G.λ * (twoπ/z0)^Rational(3, 2) * cis(dot(G.q, G.p)) * exp(-sum(abs2, G.p) / (2*z0)) * erf(sqrt(z0 / 2) * r) / r
    end
end