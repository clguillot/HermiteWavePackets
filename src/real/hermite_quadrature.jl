
#=
    Reminder :
    -hermiteh(n, x) computes Hₙ(x)
     where the (Hₙ)ₙ are the Hermite polynomials orthognal with respect to the weight exp(-x²)
    -gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,m) such that the quadrature formula
     ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1
=#


#=

    INTEGRAL

=#

#=
    Returns a vector containing the value of the integral of the N first hermite functions
=#
@generated function hermite_primitive_integral(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32, Float64}}
    
    U = zeros(T, N)

    if N > 0
        U[1] = sqrt(T(2)) * T(π)^(1/4)
        for k=3:2:N
            U[k] = sqrt(T(k - 2) / T(k - 1)) * U[k - 2]
        end
    end

    V = SizedVector{N}(U)

    return :( $V )
end

#=
    Returns a vector containing the value of the integral of the N first hermite functions
=#
function hermite_integral(a::Real, q::Real, ::Val{N}) where{N}
    T = fitting_float(promote_type(typeof(a), typeof(q)))
    U0 = hermite_primitive_integral(T, Val(N))
    return a^T(-1/4) .* SVector{N}(U0)
end


#=

    QUADRATURE RULE

=#

#=
    Computes two lists w and x of N weights and nodes such that the quadrature formula
        ∫dx f(x) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
    is exact for any function of the form f(x)exp(-x²)
=#
@generated function hermite_primitive_quadrature(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32, Float64}}
    x0, w0 = gausshermite(N)
    for j in eachindex(w0)
        w0[j] *= exp(x0[j]^2)
    end

    x0 = SizedVector{N}(T.(x0))
    w0 = SizedVector{N}(T.(w0))

    return :( $x0, $w0 )
end

#=
    Computes two lists x, w of N weights and nodes such that the quadrature formula
        ∫dx f(x)exp(-a(x-q)²) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
    is exact for any polynomial f up to degree 2N-1
=#
function hermite_quadrature(a::Real, q::Real, ::Val{N}) where{N}
    T = fitting_float(promote_type(typeof(a), typeof(q)))    
    x0, w0 = hermite_primitive_quadrature(T, Val(N))

    c = a^T(-1/2)
    x = SVector{N}(x0) .* c .+ q
    w = SVector{N}(w0) .* c

    return x, w
end


#=

    TRANSFORM

=#

#=
    Let x, w = hermite_quadrature(1, 0, N)
    This functions returns x, M where
    - x are the gausshermite quadrature points
    - M is a matrix of size N×N that transforms the values of a function
        at the quadrature points x into its discrete transform on the Fourier base
     In other words, if φ(x) = ∑ₙ λₙψₙ(1, 0, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        MΦ = Λ, with Λ=(λₙ)ₙ
=#
@generated function hermite_primitive_discrete_transform(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32, Float64}}
    x, _ = hermite_primitive_quadrature(Float64, Val(N))

    M = zeros(Float64, N, N)

    if N > 0
        b = sqrt(2)
        @views @. M[:, 1] = π^(-1/4) * exp(-x^2 / 2)
        if N > 1
            @views @. M[:, 2] = b * x * M[:, 1]

            for k=3:N
                @views @. M[:, k] = (b * x * M[:, k-1] - sqrt(k-2) * M[:, k-2]) / sqrt(k-1)
            end
        end
    end

    x = SizedVector{N}(T.(x))
    M_inv1 = SizedMatrix{N, N}(T.(M^(-1)))

    return :( $x, $M_inv1 )
end

#=
    Let x, w = hermite_integral_quadrature(a, q, N)
    This functions returns x, M where
    - x is a SVector containing the N Gauss-Hermite quadrature points
    - M is a SMatrix of size N×N that transforms the values of a function
        at the quadrature points x into its discrete transform on the Fourier base
     In other words, if φ(x) = ∑ₙ λₙψₙ(a, q, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        MΦ = Λ, with Λ=(λₙ)ₙ
=#
function hermite_discrete_transform(a::Real, q::Real, ::Val{N}) where{N}
    T = fitting_float(promote_type(typeof(a), typeof(q)))
    x0, M0 = hermite_primitive_discrete_transform(T, Val(N))

    x = SVector{N}(x0) .* a^T(-1/2) .+ q
    M = SMatrix{N, N}(M0) .* a^T(-1/4)

    return x, M
end