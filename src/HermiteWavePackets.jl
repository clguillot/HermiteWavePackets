module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using FastGaussQuadrature

# Abtract types
abstract type AbstractWavePacket{D} end
export AbstractWavePacket

# Concrete types
export Gaussian
export GaussianWavePacket
export HermiteFct
export HermiteWavePacket
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

include("gaussian_arg.jl")
include("hermite_quadrature.jl")
include("generics.jl")

include("gaussian_wave_packet.jl")
include("hermite_wave_packet.jl")

include("wave_packet_sum.jl")

end
