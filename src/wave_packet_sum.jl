#=

    STATIC ARRAY OF WAVE PACKETS

=#

#=
    Wraps a container of AbstractWavePacket into a WavePacketSum <: AbstractWavePacket
    Represents the function x -> sum(p(x) for p in g)
=#
struct WavePacketSum{Ctype<:Union{AbstractArray{<:AbstractWavePacket}, Tuple{Vararg{<:AbstractWavePacket}}}} <: AbstractWavePacket
    g::Ctype
end

#=
    CREATION
=#

function Base.:+(G1::AbstractWavePacket, G2::AbstractWavePacket)
    return WavePacketSum((G1, G2))
end

function Base.:-(G1::AbstractWavePacket, G2::AbstractWavePacket)
    return WavePacketSum((G1, -G2))
end

#=
    BASIC OPERATIONS
=#

#
function core_type(::Type{WavePacketSum{Ctype}}) where{Ctype<:AbstractArray}
    return core_type(eltype(Ctype))
end
function core_type(::Type{WavePacketSum{Ctype}}) where{Ctype<:Tuple}
    return promote_type(core_type.(fieldtypes(Ctype))...)
end

#
function fitting_float(::Type{WavePacketSum{Ctype}}) where Ctype
    return fitting_float(core_type(Ctype))
end

#
function Base.conj(G::WavePacketSum)
    return WavePacketSum(conj.(G.g))
end

#=
    Computes ∑ₖ G[k](x)
=#
function (G::WavePacketSum)(x)
    s = zero(promote_type(core_type(G), eltype(x)))
    return sum(g(x) for g in G.g; init=s)
end

#=
    TRANSFORMATIONS
=#

#
function Base.:-(G1::WavePacketSum)
    return WavePacketSum(.- G1.g)
end

#
function Base.:*(w::Number, G::WavePacketSum)
    return WavePacketSum(w .* G.g)
end

#
function Base.:*(G1::AbstractWavePacket, G2::WavePacketSum)
    return WavePacketSum(G1 .* G2.g)
end
@inline function Base.:*(G1::WavePacketSum, G2::AbstractWavePacket)
    return G2 * G1
end

#
function convolution(G1::AbstractWavePacket, G2::WavePacketSum)
    return WavePacketSum(convolution.(G1, G2.g))
end
@inline function convolution(G1::WavePacketSum, G2::AbstractWavePacket)
    return convolution(G2, G1)
end

#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::WavePacketSum)
    s = zero(core_type(G))
    return sum(integral(g) for g in G.g; init=s)
end

#=
    Computes the dot product
        ∑ₖ,ₗ dot_L2(G1[k], G2[l])
=#
function dot_L2(G1::WavePacketSum, G2::WavePacketSum)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(dot_L2(g1, g2) for g1 in G1.g for g2 in G2.g; init=s)
end

#=
    Computes the dot product
        ∑ₗ dot_L2(G1, G2[l])
=#
function dot_L2(G1::AbstractWavePacket, G2::WavePacketSum)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(dot_L2(G1, g2) for g2 in G2.g; init=s)
end
#=
    Computes the dot product
        ∑ₖ dot_L2(G1[k], G2)
=#
@inline function dot_L2(G1::WavePacketSum, G2::AbstractWavePacket)
    return dot_L2(G2, G1)
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::WavePacketSum)
    s = zero(real(core_type(G)))
    S1 = sum(norm2_L2(g) for g in G.g; init=s)
    S2 = sum(real(dot_L2(G.g[k], G.g[l])) for k in eachindex(G.g) for l in Iterators.drop(eachindex(G.g), k); init=s)
    return S1 + 2 * S2
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
        sub_state = iterate(sup_state)
        if !isnothing(sub_state)
            return (sub_state[1], (sup_state..., sub_state[2]))
        end
        sup_state = iterate(G.g, sup_state[2])
    end
    return nothing
end
function Base.iterate(G::WavePacketSum, state)
    sup_state = (state[1], state[2])
    sub_state = iterate(sup_state, state[3])

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