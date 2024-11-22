module ComplexHermiteFct

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

# Custom mathematical operations compatible with autodifferentiation
include("math.jl")

@inline function fitting_float(::Type{T}) where{T<:Number}
    prec = precision(real(T))
    if prec == precision(Float16)
        Float16
    elseif prec == precision(Float32)
        Float32
    elseif prec == precision(Float64)
        Float64
    else
        throw(ArgumentError("The precision format of $T is unsupported"))
    end
end

include("real/gaussian1d.jl")
include("real/hermite_fct1d.jl")
include("real/hermite_quadrature.jl")

include("complex/complex_gaussian1d.jl")
include("complex/complex_hermite_fct1d.jl")

end
