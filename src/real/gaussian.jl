
#=
    Represents the gaussian function
        λ*exp(-∑ₖ aₖ/2*(xₖ-qₖ)²)
=#
const Gaussian{D, Tλ<:Number, Tz<:Real, Tq<:Real} = GaussianWavePacket{D, Tλ, Tz, Tq, NullNumber}

function Gaussian(λ::Number, z::SVector{D, <:Real}, q::SVector{D, <:Real}) where D
    return GaussianWavePacket(λ, z, q, zeros(SVector{D, NullNumber}))
end

# Computes the L² product of two real gaussians
function dot_L2(G1::Gaussian{D, Tλ1, Ta1, Tq1}, G2::Gaussian{D, Tλ2, Ta2, Tq2}) where{D, Tλ1<:Real, Ta1, Tq1, Tλ2, Ta2, Tq2}
    return integral(G1 * G2)
end