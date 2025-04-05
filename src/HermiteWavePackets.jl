module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

# Abtract types
abstract type AbstractWavePacket{D} end
export AbstractWavePacket

# Concrete types
export Gaussian1D
export HermiteFct1D
export Gaussian
export HermiteFct
export GaussianWavePacket1D
export HermiteWavePacket1D
export GaussianWavePacket
export WavePacketSum

# Functions
export hermite_grid
export evaluate_grid
export hermite_discrete_transform
export hermite_quadrature
export truncate_to_gaussian
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

include("utils.jl")
include("nullnumber.jl")

include("real/hermite_quadrature.jl")

include("complex/gaussian_wave_packet.jl")
include("complex/hermite_wave_packet.jl")

include("real/hermite_fct_1d.jl")
include("real/hermite_fct.jl")

include("complex/hermite_wave_packet_1d.jl")

include("wave_packet_sum.jl")
include("generics.jl")

end
