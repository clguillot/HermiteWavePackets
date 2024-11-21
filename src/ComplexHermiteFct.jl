module ComplexHermiteFct

using StaticArrays
using FastGaussQuadrature

# Custom mathematical operations compatible with autodifferentiation
include("math.jl")

@inline function fitting_float(::Type{T}) where{T<:Real}
    prec = max(precision(a), precision(q))
    if prec <= precision(Float16)
        Float16
    elseif prec <= precision(Float32)
        Float32
    else
        Float64
    end
end

include("real/gaussian1d.jl")
include("real/hermite_fct1d.jl")
include("real/hermite_quadrature.jl")

include("complex/complex_gaussian1d.jl")
include("complex/complex_hermite_fct1d.jl")

end
