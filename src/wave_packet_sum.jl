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

#=
    Computes ∑ₖ G[k](x)
=#
function (G::WavePacketSum)(x)
    s = zero(promote_type(core_type(G), eltype(x)))
    for g in G.g
        s += g(x)
    end
    return s
end

#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::WavePacketSum)
    s = zero(core_type(G))
    for g in G.g
        s += integral(g)
    end
    return s
end

#=
    Computes the dot product
        ∑ₖ,ₗ dot_L2(G1[k], G2[l])
=#
function dot_L2(G1::WavePacketSum, G2::WavePacketSum)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g1 in G1.g
        for g2 in G2.g
            s += dot_L2(g1, g2)
        end
    end
    return s
end

#=
    Computes the dot product
        ∑ₖ dot_L2(G1[k], G2)
=#
function dot_L2(G1::WavePacketSum, G2::AbstractWavePacket)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g1 in G1.g
        s += dot_L2(g1, G2)
    end
    return s
end

#=
    Computes the dot product
        ∑ₗ dot_L2(G1, G2[l])
=#
function dot_L2(G1::AbstractWavePacket, G2::WavePacketSum)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g2 in G2.g
        s += dot_L2(G1, g2)
    end
    return s
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::WavePacketSum)
    s = zero(real(core_type(G)))
    for k in eachindex(G.g)
        s += norm2_L2(G.g[k])
        for l in Iterators.drop(eachindex(G.g), k)
            s += 2 * real(dot_L2(G.g[k], G.g[l]))
        end
    end
    return s
end