
function Base.zero(::T) where{T<:AbstractWavePacket}
    return zero(T)
end

function core_type(::T) where{T<:AbstractWavePacket}
    return core_type(T)
end

function core_type(::Type{T}) where{T<:Union{Number, NullNumber}}
    return T
end
function core_type(::T) where{T<:Union{Number, NullNumber}}
    return core_type(T)
end

function core_type(S, T)
    return promote_type(core_type(S), core_type(T))
end
function core_type(S, T, U...)
    return promote_type(core_type(S), core_type(T), core_type(U...))
end

@generated function fitting_float(::Type{T}) where{T<:Number}
    prec = precision(real(T))
    if prec == precision(Float16)
        return :( Float16 )
    elseif prec == precision(Float32)
        return :( Float32 )
    elseif prec == precision(Float64)
        return :( Float64 )
    else
        throw(ArgumentError("The precision format of $T is unsupported"))
    end
end
function fitting_float(T...)
    return fitting_float(core_type(T...))
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