#
function core_type(::Type{AbstractVector{TG}}) where{TG<:AbstractWavePacket}
    return core_type(TG)
end
function core_type(G::AbstractVector{TG}) where{TG<:AbstractWavePacket}
    return core_type(TG)
end

#=
    Computes ∑ₖ G[k](x)
=#
function (G::AbstractVector{<:AbstractWavePacket})(x::T) where{T<:Number}
    s = zero(promote_type(core_type(G), T))
    for g in G
        s += g(x)
    end
    return s
end

#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::AbstractVector{<:AbstractWavePacket})
    s = zero(core_type(G))
    for g in G
        s += integral(g)
    end
    return s
end

#=
    Computes the dot product
        ∑ₖ,ₗ dot_L2(G1[k], G2[l])
=#
function dot_L2(G1::AbstractVector{<:AbstractWavePacket}, G2::AbstractVector{<:AbstractWavePacket})
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g1 in G1
        for g2 in G2
            s += dot_L2(g1, g2)
        end
    end
    return s
end

#=
    Computes the dot product
        ∑ₖ dot_L2(G1[k], G2)
=#
function dot_L2(G1::AbstractVector{<:AbstractWavePacket}, G2::AbstractWavePacket)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g1 in G1
        s += dot_L2(g1, G2)
    end
    return s
end

#=
    Computes the dot product
        ∑ₗ dot_L2(G1, G2[l])
=#
function dot_L2(G1::AbstractWavePacket, G2::AbstractVector{<:AbstractWavePacket})
    s = zero(promote_type(core_type(G1), core_type(G2)))
    for g2 in G2
        s += dot_L2(G1, g2)
    end
    return s
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::AbstractVector{<:AbstractWavePacket})
    s = real(zero(core_type(G)))
    for k in eachindex(G)
        s += norm2_L2(G[k])
        for l in k+1:length(G)
            s += 2 * real(dot_L2(G[k], G[l]))
        end
    end
    return s
end

#=
    Computes the L2 norm of ∑ₖ G[k]
=#
function norm_L2(G::AbstractVector{<:AbstractWavePacket})
    return sqrt(norm2_L2(G))
end