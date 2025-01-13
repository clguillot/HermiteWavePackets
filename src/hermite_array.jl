#=
    Computes ∑ₖ G[k](x)
=#
function (G::AbstractVector{<:AbstractWavePacket})(x::Number)
    s = zero(promote_type(eltype(eltype(G)), typeof(x)))
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
    s = zero(eltype(eltype(G)))
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
    s = zero(promote_type(eltype(eltype(G1)), eltype(eltype(G1))))
    for g1 in G1
        for g2 in G2
            s += dot_L2(g1, g2)
        end
    end
    return s
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::AbstractVector{<:AbstractWavePacket})
    s = real(zero(eltype(eltype(G))))
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