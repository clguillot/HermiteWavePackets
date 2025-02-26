
function zero(::T) where{T<:AbstractWavePacket}
    return zero(T)
end

function core_type(::T) where{T<:AbstractWavePacket}
    return core_type(T)
end

function fitting_float(::T) where{T<:AbstractWavePacket}
    return fitting_float(T)
end

function centers(G::AbstractWavePacket)
    return SVector{1}(center(G))
end

@inline function (*)(G::AbstractWavePacket, w::Number)
    return w * G
end

@inline function norm_L2(G::AbstractWavePacket)
    return sqrt(norm2_L2(G))
end