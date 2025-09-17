#=

    SUM OF WAVE PACKETS

=#

struct WavePacketSum{D, Ctype} <: AbstractWavePacket{D}
    g::Ctype

    # Constrain Ctype using an inner constructor
    function WavePacketSum{D}(g::Ctype) where{D, Ctype}
        if !all(G -> typeof(G) <: AbstractWavePacket{D}, g)
            throw(ArgumentError("All elements of g must be subtypes of AbstractWavePacket{$D}"))
        end
        return new{D, Ctype}(g)
    end
    function WavePacketSum{D}(g::AbstractWavePacket{D}...) where D
        return WavePacketSum{D}(g)
    end
    function WavePacketSum(g::AbstractWavePacket{D}...) where D
        return WavePacketSum{D}(g)
    end
end


#=
    CREATION
=#

function Base.:+(G::AbstractWavePacket{D}...) where D
    return WavePacketSum{D}(G)
end

function Base.:-(G1::AbstractWavePacket{D}, G2::AbstractWavePacket{D}) where D
    return WavePacketSum{D}(G1, -G2)
end

#=
    BASIC OPERATIONS
=#

#
function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:AbstractArray}
    return core_type(eltype(Ctype))
end
function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple}
    return promote_type(core_type.(fieldtypes(Ctype))...)
end

#
function Base.conj(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(conj.(G.g))
end

#=
    Computes ∑ₖ G[k](x)
=#
function (G::WavePacketSum)(x::AbstractVector{<:Union{Number, NullNumber}})
    s = zero(promote_type(core_type(G), eltype(x)))
    return sum(g -> g(x), G.g; init=s)
end

#=
    TRANSFORMATIONS
=#

#
function Base.:-(G1::WavePacketSum{D}) where D
    return WavePacketSum{D}(.- G1.g)
end

#
function Base.:*(w::Number, G::WavePacketSum{D}) where D
    return WavePacketSum{D}(w .* G.g)
end

#
function Base.:*(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g1 -> g1 * G2, G1.g))
end
function Base.:*(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g2 -> G1 * g2, G2.g))
end
function Base.:*(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    return G2 * G1
end

#
function polynomial_product(G::WavePacketSum{D}, P::SArray{NP, <:Number, D},
                q::SVector{D, <:Union{Real, NullNumber}}=zeros(SVector{D, NullNumber})) where{D, NP}
    return WavePacketSum{D}(broadcast(g -> polynomial_product(g, P, q), G.g))
end

#
function unitary_product(G::WavePacketSum{D}, b, q=zeros(SVector{D, NullNumber}), p=zeros(SVector{D, NullNumber})) where D
    return WavePacketSum{D}(broadcast(g -> unitary_product(g, b, q, p), G.g))
end

#
function convolution(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g1 -> convolution(g1, G2), G1.g))
end
function convolution(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g2 -> convolution(G1, g2), G2.g))
end
function convolution(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    return G2 * G1
end



#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::WavePacketSum)
    s = zero(core_type(G))
    return sum(g -> integral(g), G.g; init=s)
end

#
function dot_L2(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    f(g1, g2) = dot_L2(g1, g2)
    return sum(gg -> f(gg...), Iterators.product(G1.g, G2.g); init=s)
end
function dot_L2(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g2 -> dot_L2(G1, g2), G2.g; init=s)
end
function dot_L2(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_L2(g1, G2), G1.g; init=s)
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::WavePacketSum)
    s = zero(real(core_type(G)))
    S1 = sum(g -> norm2_L2(g), G.g; init=s)
    f(k, l) = k < l ? real(dot_L2(G.g[k], G.g[l])) : s
    S2 = sum(i -> f(i...), Iterators.product(eachindex(G.g), eachindex(G.g)); init=s)
    return S1 + 2 * S2
end



#
function dot_∇(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    f(g1, g2) = dot_∇(g1, g2)
    return sum(gg -> f(gg...), Iterators.product(G1.g, G2.g); init=s)
end
function dot_∇(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g2 -> dot_∇(G1, g2), G2.g; init=s)
end
function dot_∇(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_∇(g1, G2), G1.g; init=s)
end

#=
    Computes the squared homogeneous H1 norm of ∑ₖ G[k]
=#
function norm2_∇(G::WavePacketSum)
    s = zero(real(core_type(G)))
    S1 = sum(g -> norm2_∇(g), G.g; init=s)
    f(k, l) = k < l ? real(dot_∇(G.g[k], G.g[l])) : s
    S2 = sum(i -> f(i...), Iterators.product(eachindex(G.g), eachindex(G.g)); init=s)
    return S1 + 2 * S2
end

# Fourier transform
function fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(fourier.(G.g))
end
# Inverse Fourier transgorm
function inv_fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(inv_fourier.(G.g))
end

#=
    ITERATOR
    It is allowed to iterate over a WavePacketSum
    The iteration will be performed over all elements of the sum,
    by breaking down internal WavePacketSum
=#
function Base.iterate(G::WavePacketSum)
    sup_state = iterate(G.g)
    while !isnothing(sup_state)
        sub_state = iterate(sup_state[1])
        if !isnothing(sub_state)
            return (sub_state[1], (sup_state..., sub_state[2]))
        end
        sup_state = iterate(G.g, sup_state[2])
    end
    return nothing
end
function Base.iterate(G::WavePacketSum, state)
    sup_state = (state[1], state[2])
    sub_state = iterate(sup_state[1], state[3])

    # If an iterated element is found on the current element
    if !isnothing(sub_state)
        return (sub_state[1], (sup_state..., sub_state[2]))
    end

    # If no iterated element is found on the current element
    sup_state = iterate(G.g, sup_state[2])
    while !isnothing(sup_state)
        sub_state = iterate(sup_state[1])
        if !isnothing(sub_state)
            return (sub_state[1], (sup_state..., sub_state[2]))
        end
        sup_state = iterate(G.g, sup_state[2])
    end

    return nothing
end