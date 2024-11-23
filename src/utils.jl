
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

@generated function fitting_float(x::T) where{T<:Real}
    Tf = fitting_float(T)
    return :( $Tf )
end

@generated function fitting_float(x::AbstractArray{T}) where{T<:Number}
    Tf = fitting_float(T)
    return :( $Tf )
end

# Returns real(x) if T<:Real, and x if T<:Complex
@generated function complex_truncation(::Type{T}, x::Tx) where{T<:Number, Tx<:Number}
    if T<:Real
        return :( real(x) )
    elseif T<:Complex
        return :( x )
    else
        :( throw(ArgumentError("T is not a Real or a Complex type")) )
    end
end