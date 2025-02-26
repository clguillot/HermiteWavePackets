#=

    STATIC ARRAY OF WAVE PACKETS

=#
struct WavePacketArray{ArrayType<:AbstractArray{<:AbstractWavePacket}} <: AbstractWavePacket
    g::ArrayType
end

#
function core_type(::Type{WavePacketArray{ArrayType}}) where ArrayType
    return core_type(eltype(ArrayType))
end

#=
    Computes ∑ₖ G[k](x)
=#
function (G::WavePacketArray)(x::T) where{T<:Number}
    s = zero(promote_type(core_type(G), T))
    for g in G.g
        s += g(x)
    end
    return s
end

#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::WavePacketArray)
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
function dot_L2(G1::WavePacketArray, G2::WavePacketArray)
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
function dot_L2(G1::WavePacketArray, G2::AbstractWavePacket)
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
function dot_L2(G1::AbstractWavePacket, G2::WavePacketArray)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g2 in G2.g
        s += dot_L2(G1, g2)
    end
    return s
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::WavePacketArray)
    s = zero(real(core_type(G)))
    for k in eachindex(G.g)
        s += norm2_L2(G.g[k])
        for l in k+1:length(G.g)
            s += 2 * real(dot_L2(G[k], G[l]))
        end
    end
    return s
end