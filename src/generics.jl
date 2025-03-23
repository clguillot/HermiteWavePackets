
function Base.zero(::T) where{T<:AbstractWavePacket}
    return zero(T)
end

function core_type(::T) where{T<:AbstractWavePacket}
    return core_type(T)
end

function core_type(x...)
    return promote_type(core_type.(x)...)
end

function fitting_float(::Type{T}) where{T<:AbstractWavePacket}
    return fitting_float(core_type(T))
end

function fitting_float(::T) where{T<:AbstractWavePacket}
    return fitting_float(T)
end

function fitting_float(x...)
    return promote_type(fitting_float.(x)...)
end

@inline function Base.:*(G::AbstractWavePacket, w::Number)
    return w * G
end

function Base.iterate(G::AbstractWavePacket)
    return (G, nothing)
end
function Base.iterate(::AbstractWavePacket, ::Nothing)
    return nothing
end

@inline function norm_L2(G::AbstractWavePacket)
    return sqrt(norm2_L2(G))
end