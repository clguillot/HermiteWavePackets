module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using DoubleFloats
using FastGaussQuadrature

# Custom mathematical operations compatible with autodifferentiation
include("utils.jl")

# Structures
export Gaussian1D
export HermiteFct1D
export GaussianWavePacket1D
export HermiteWavePacket1D

# Functions
export evaluate
export integral
export unitary_product
export fourier
export inv_fourier
export convolution
export dot_L2

import Base.*
import Base.copy
import Base.zero
import Base.conj

include("real/gaussian1d.jl")
include("real/hermite_fct_1d.jl")
include("real/hermite_quadrature.jl")

include("complex/gaussian_wave_packet_1d.jl")
include("complex/hermite_wave_packet_1d.jl")

end
