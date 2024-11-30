
#=
    Reminder :
    - gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,m) such that the quadrature formula
     ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1
=#


#=

    TRANSFORM

=#

#=
    This functions returns x, w, M where
    - x, w are such that the quadrature rule
            ∫dx f(x) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
        is exact for any function of the form f(x)exp(-x²)
    - M is a matrix of size N×N that transforms the values of a function
        at the quadrature points x into its discrete transform on the Fourier base
     In other words, if φ(x) = ∑ₙ λₙψₙ(1, 0, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        MΦ = Λ, with Λ=(λₙ)ₙ
=#
@generated function hermite_primitive_discrete_transform(::Type{Float64}, ::Val{N}) where{N}
    x, _ = gausshermite(N)

    x = Double64.(x)
    L = zeros(Double64, N, N)

    if N > 0
        b = sqrt(Double64(2))
        @views @. L[:, 1] = Double64(π)^(-1/Double64(4)) * exp(-x^2 / 2)
        if N > 1
            @views @. L[:, 2] = b * x * L[:, 1]

            for k=3:N
                @views @. L[:, k] = (b * x * L[:, k-1] - sqrt(Double64(k-2)) * L[:, k-2]) / sqrt(Double64(k-1))
            end
        end
    end

    M = L^(-1)
    w = @views [dot(M[:, k], M[:, k]) for k in 1:N]

    x = SizedVector{N}(Float64.(x))
    w = SizedVector{N}(Float64.(w))
    M = SizedMatrix{N, N}(Float64.(M))

    return :( $x, $w, $M )
end
@generated function hermite_primitive_discrete_transform(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32}}
    x64, w64, M64 = hermite_primitive_discrete_transform(Float64, Val(N))

    x = SizedVector{N}(T.(x64))
    w = SizedVector{N}(T.(w64))
    M = SizedMatrix{N, N}(T.(M64))

    return :( $x, $w, $M )
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
    x0, _, M0 = hermite_primitive_discrete_transform(T, Val(N))

    x = SVector{N}(x0) .* a^T(-1/2) .+ q
    M = SMatrix{N, N}(M0) .* a^T(-1/4)

    return x, M
end


#=

    QUADRATURE RULE

=#

#=
    Computes two lists x, w of N weights and nodes such that the quadrature formula
        ∫dx f(x)exp(-a(x-q)²) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
    is exact for any polynomial f up to degree 2N-1
=#
function hermite_quadrature(a::Real, q::Real, ::Val{N}) where{N}
    T = fitting_float(promote_type(typeof(a), typeof(q)))    
    x0, w0, _ = hermite_primitive_discrete_transform(T, Val(N))

    c = a^T(-1/2)
    x = SVector{N}(x0) .* c .+ q
    w = SVector{N}(w0) .* c

    return x, w
end