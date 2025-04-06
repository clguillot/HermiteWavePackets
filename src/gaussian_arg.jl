#=
    Computes z, q, p such that exp(-z/2*(x-q)²)*exp(ipx) is equal
    to the product
        C * exp(-z1/2*(x-q1)²)*exp(ip1*x) * exp(-z2/2*(x-q2)²)*exp(ip2*x)
    where C is some constant
=#
function gaussian_product_arg(z1::SVector{D, <:Number}, q1::SVector{D, <:Union{Real, NullNumber}}, p1::SVector{D, <:Union{Real, NullNumber}},
                z2::SVector{D, <:Number}, q2::SVector{D, <:Union{Real, NullNumber}}, p2::SVector{D, <:Union{Real, NullNumber}}) where D
    z = @. z1 + z2
    q = @. (real(z1) * q1 + real(z2) * q2) / (real(z1) + real(z2))
    p0 = @. (imagz(z1) * q1 + imagz(z2) * q2) - (imagz(z1) + imagz(z2)) * q
    p = @. p2 + p1 + p0
    return z, q, p
end

#=
    Computes z_tf, q_tf, p_tf such that exp(-z_tf/2*(x-q_tf)²)*exp(ip_tf*x) is equal
    to the Fourier transform of
        C * exp(-z/2*(x-q)²)*exp(ip*x)
    where C is some constant
=#
function complex_gaussian_fourier_arg(z, q, p)
    return @. inv(z), p, -q
end

#=
    Computes z_itf, q_itf, p_itf such that exp(-z_itf/2*(x-q_itf)²)*exp(ip_itf*x) is equal
    to the inverse Fourier transform of
        C * exp(-z/2*(x-q)²)*exp(ip*x)
    where C is some constant
=#
function complex_gaussian_inv_fourier_arg(z, q, p)
    return @. inv(z), -p, q
end

#=
    Computes z, q, p such that exp(-z/2*(x-q)²)*exp(ipx) is equal
    to the convolution product
        C * exp(-z1/2*(x-q1)²)*exp(ip1*x) ∗ exp(-z2/2*(x-q2)²)*exp(ip2*x)
    where C is some constant
=#
function complex_gaussian_convolution_product_arg(z1, q1, p1, z2, q2, p2)
    z1_tf, q1_tf, p1_tf = complex_gaussian_fourier_arg(z1, q1, p1)
    z2_tf, q2_tf, p2_tf = complex_gaussian_fourier_arg(z2, q2, p2)
    z_tf, q_tf, p_tf = complex_gaussian_product_arg(z1_tf, q1_tf, p1_tf, z2_tf, q2_tf, p2_tf)
    return complex_gaussian_inv_fourier_arg(z_tf, q_tf, p_tf)
end

#=
    Computes a, q such that exp(-a/2*(x-q)²) is equal (up to some constant)
    to the convolution product
        exp(-a1/2*(x-q1)²) ∗ exp(-a2/2*(x-q2)²)
=#
function gaussian_convolution_arg(a1, q1, a2, q2)
    a = @. a1 * a2 / (a1 + a2)
    q = @. q1 + q2
    return a, q
end