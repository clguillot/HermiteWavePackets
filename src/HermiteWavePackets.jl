module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

# Custom mathematical operations compatible with autodifferentiation
include("utils.jl")

# Abtract types
abstract type AbstractWavePacket end
abstract type AbstractWavePacket1D <: AbstractWavePacket end
export AbstractWavePacket
export AbstractWavePacket1D

# Concrete types
export Gaussian1D
export HermiteFct1D
export GaussianWavePacket1D
export HermiteWavePacket1D

# Functions
export hermite_discrete_transform
export hermite_quadrature
export integral
export unitary_product
export polynomial_product
export fourier
export inv_fourier
export convolution
export dot_L2
export norm2_L2
export norm_L2

import Base.*
import Base.convert
import Base.copy
import Base.zero
import Base.conj
import Base.eltype

include("real/gaussian1d.jl")
include("real/hermite_fct_1d.jl")
include("real/hermite_quadrature.jl")

include("complex/gaussian_wave_packet_1d.jl")
include("complex/hermite_wave_packet_1d.jl")

include("hermite_array.jl")

end
