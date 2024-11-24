using ComplexHermiteFct

using LinearAlgebra
using FastGaussQuadrature
using SpecialFunctions
using StaticArrays

function hermite_poly(n::Integer, x::T) where{T<:Number}
    v = zero(T)
    u = one(T)
    for j in 1:n
        w = u
        u = 2*x*u - 2*(j-1)*v
        v = w
    end
    return u
end

include("test_gaussian.jl");
include("test_hermite.jl");

include("test_hermite_wave_packet.jl")
# include("test_hermite_wave_packet.jl")