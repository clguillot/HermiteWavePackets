# Complex exponential compatible with autodiff
@inline function myexp(x::Real)
    return exp(x)
end
@inline function myexp(z::Complex)
    er = exp(real(z))
    s, c = sincos(imag(z))
    return complex(er * c, er * s)
end


# Imaginary exponential compatible with autodiff
@inline function mycis(x::Real)
    s, c = sincos(x)
    return complex(c, s)
end
@inline function mycis(z::Complex)
    er = exp(-imag(z))
    s, c = sincos(real(z))
    return complex(er * c, er * s)
end


# Complex square root compatible with autodiff
@inline function mysqrt(x::Real)
    return sqrt(x)
end
@inline function mysqrt(z::Complex)
    x = real(z)
    y = imag(z)
    r = hypot(x, y)
    s = sqrt((r + x) / 2)
    return complex(s, y / (2*s))
end