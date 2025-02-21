module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

# Abtract types
abstract type AbstractWavePacket end
abstract type AbstractWavePacket1D <: AbstractWavePacket end
export AbstractWavePacket
export AbstractWavePacket1D

# Concrete types
export Gaussian1D
export HermiteFct1D
export Gaussian
export GaussianWavePacket1D
export HermiteWavePacket1D
export GaussianWavePacket

# Functions
export hermite_discrete_transform
export hermite_quadrature
export integral
export evaluate
export unitary_product
export polynomial_product
export fourier
export inv_fourier
export convolution
export dot_L2
export norm2_L2
export norm_L2
export core_type

import Base.*
import Base.convert
import Base.copy
import Base.zero
import Base.conj

include("utils.jl")

include("real/hermite_quadrature.jl")

include("real/gaussian1d.jl")
include("real/hermite_fct_1d.jl")
include("real/gaussian.jl")

include("complex/gaussian_wave_packet_1d.jl")
include("complex/hermite_wave_packet_1d.jl")

include("hermite_array.jl")
include("generics.jl")

end
