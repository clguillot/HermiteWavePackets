
# Returns real(x) if T<:Real, and x if T<:Complex
@generated function complex_truncation(::Type{T}, x::Tx) where{T<:Number, Tx<:Number}
    if T<:Real
        return :( real(x) )
    elseif T<:Complex
        return :( x )
    else
        :( throw(ArgumentError("T is not a Real or a Complex type")) )
    end
end

#=

    CLENSHAW ALGORITHM FOR HERMITE FUNCTIONS

=#

#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, x)
    where
        α₀ = α0
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::Tx, α0::Tα) where{TΛ<:Number, Tx<:Number, Tz<:Number, Tα<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Tα))
    N = length(Λ)
    u = zero(promote_type(TΛ, Tz, Tx, T))
    if N > 0
        u += Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = Λ[N-1] + zx * u / sqrt(T(N-1))
            for k=N-3:-1:0
                (u, v) = (Λ[k+1] + zx * u / sqrt(T(k+1)) - sqrt(T(k+1) / T(k+2)) * v, u)
            end
        end
    end
    return u * α0
end
#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, xₘ) (m=1,...,M)
    where
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::SVector{M, Tx}, α0::SVector{M, Tα}) where{TΛ<:Number, M, Tx<:Number, Tz<:Number, Tα<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Tα))
    N = length(Λ)
    u = zero(SVector{M, promote_type(TΛ, Tz, Tx, T)})
    if N > 0
        u = u .+ Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = @. Λ[N-1] + zx * u / sqrt(T(N-1))
            for k=N-3:-1:0
                (u, v) = @. (Λ[k+1] + zx * u / sqrt(T(k+1)) - sqrt(T(k+1) / T(k+2)) * v, u)
            end
        end
    end
    return u .* α0
end

#=
    Computes the clenshaw algorithm along dimension dim:
        ∑ₙ λₙ αₙ(z, x)
    where
        α₀ = α
        αₙ₊₁ = (x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
    By default, performs the transformation along the last axis of Λ
=#
@generated function clenshaw_hermite_transform(Λ::SArray{N, TΛ}, x::Tx, μ::Tμ, α::Tα, ::Val{dim}=Val(lastindex(axes(Λ)))) where{N, TΛ<:Number, Tx, Tμ, Tα, dim}
    TC = promote_type(TΛ, Tx, Tμ)
    T = fitting_float(TC)
    S = size(Λ)
    if !(dim in eachindex(S))
        throw(ArgumentError("Trying to perform a transform along dimension $dim which does not exist in Λ"))
    end
    kdim = length(S[begin:dim])
    array_access_drop = Meta.parse("Λ[" * prod(":," for k in 1:kdim-1; init="") * "k," * prod(":," for k in kdim+1:ndims(Λ); init="") * "]")
    sub_size = Tuple{S[begin:dim-1]..., S[dim+1:end]...}
    t_size = size(Λ, dim)
    code =
        quote
            u = zero(SArray{$sub_size, $TC})
            v = zero(SArray{$sub_size, $TC})
            for (n, k) in zip($t_size-1:-1:0, reverse(axes(Λ, $dim)))
                (u, v) =
                    ($TC.($array_access_drop .+ (x / sqrt($T(n+1))) * u .- (μ * sqrt($T(n+1) / $T(n+2))) * v), u)
            end
            return α * u
        end
    return code
end
#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, x)
    where
        α₀ = α0
        αₙ₊₁ = (z*x*αₙ(z, x) - μ * √n*αₙ₋₁(z, x)) / √(n+1)
=#
@generated function clenshaw_hermite_transform_grid(Λ::SArray{N, TΛ, D}, z::SVector{D}, x::Tuple, μ::SVector{D}, α0::Tuple) where{N, D, TΛ}
    if length(x.parameters) != D
        throw(DimensionMismatch("Expecting the length of x to be $D, but got $(length(x.parameters)) instead"))
    end
    if length(α0.parameters) != D
        throw(DimensionMismatch("Expecting the length of α0 to be $D, but got $(length(α0.parameters)) instead"))
    end
    for (y, α) in zip(x.parameters, α0.parameters)
        if !(y<:SVector)
            throw(ArgumentError("All elements of `x` must be `SVector`"))
        end
        if !(α<:SVector)
            throw(ArgumentError("All elements of `α0` must be `SVector`"))
        end
        if length(y) != length(α)
            throw(DimensionMismatch("Elements of x and α0 must have compatible size"))
        end
    end
    
    if D > 0
        zs = zero(SVector{length(x.parameters[end]), Bool})
        zt = tuple((false for _ in zs)...)
        expr1 = [:( clenshaw_hermite_transform(Λ, z[end]*x[end][$k], μ[end], α0[end][$k]) ) for k in eachindex(zs) ]
        expr2 = [:( clenshaw_hermite_transform_grid(A[$k], z_red, x_red, μ_red, α0_red) ) for k in eachindex(zt)]
        expr3 = [:( reshape(B[$k], Size(length(B[$k]))) ) for k in eachindex(zt)]
        new_size = Size((length(y) for y in x.parameters)...)
        code =
            quote
                z_red = SVector{D-1}(@view z[begin:end-1])
                x_red = Base.front(x)
                μ_red = SVector{D-1}(@view μ[begin:end-1])
                α0_red = Base.front(α0)
                A = tuple($(expr1...))
                B = tuple($(expr2...))
                return reshape(vcat($(expr3...)), $new_size)
            end
        return code
    else
        return :( return Λ )
    end
end

#=

    HORNER ALGORITHM

=#

#=
    Computes the horner algorithm along dimension dim:
        ∑ₙ λₙ αₙ(x)
    where
        α₀(x) = 1
        αₙ₊₁(x) = x*αₙ(x)
    By default, performs the transformation along the last axis of Λ
=#
@generated function horner_transform(Λ::SArray{N, TΛ}, x::Tx, ::Val{dim}=Val(lastindex(axes(Λ)))) where{N, TΛ, Tx, dim}
    TC = promote_type(TΛ, Tx)
    S = size(Λ)
    if !(dim in eachindex(S))
        throw(ArgumentError("Trying to perform a transform along dimension $dim which does not exist in Λ"))
    end
    kdim = length(S[begin:dim])
    array_access_drop = Meta.parse("Λ[" * prod(":," for k in 1:kdim-1; init="") * "k," * prod(":," for k in kdim+1:ndims(Λ); init="") * "]")
    sub_size = Tuple{S[begin:dim-1]..., S[dim+1:end]...}
    code =
        quote
            u = zero(SArray{$sub_size, $TC})
            for k in reverse(axes(Λ, $dim))
                u = $TC.($array_access_drop .+ x .* u)
            end
            return u
        end
    return code
end
#=
    Computes the horner algorithm along dimension dim:
        ∑ₙ λₙ αₙ(x)
    where
        α₀(x) = α0
        αₙ₊₁(x) = λ*(x-q)*αₙ(x)
    By default, performs the transformation along the last axis of Λ
=#
@generated function horner_transform_grid(Λ::SArray{N, <:Number, D}, λ::SVector{D}, q::SVector{D}, x::Tuple) where{N, D}
    if length(x.parameters) != D
        throw(DimensionMismatch("Expecting the length of x to be $D, but got $(length(x.parameters)) instead"))
    end
    for y in x.parameters
        if !(y<:SVector)
            throw(ArgumentError("All elements of `x` must be `SVector`"))
        end
    end
    
    if D > 0
        zs = zero(SVector{length(x.parameters[end]), Bool})
        zt = tuple((false for _ in zs)...)
        expr1 = [:( horner_transform(Λ, λ[end]*(x[end][$k] - q[end])) ) for k in eachindex(zs) ]
        expr2 = [:( horner_transform_grid(A[$k], λ_red, q_red, x_red) ) for k in eachindex(zt)]
        expr3 = [:( reshape(B[$k], Size(length(B[$k]))) ) for k in eachindex(zt)]
        new_size = Size((length(y) for y in x.parameters)...)
        code =
            quote
                λ_red = SVector{D-1}(@view λ[begin:end-1])
                q_red = SVector{D-1}(@view q[begin:end-1])
                x_red = Base.front(x)
                A = tuple($(expr1...))
                B = tuple($(expr2...))
                return reshape(vcat($(expr3...)), $new_size)
            end
        return code
    else
        return :( return Λ )
    end
end

#=
    STATIC TENSOR TRANSFORM
=#
@generated function static_tensor_transform(A::SArray{N, Ta, D}, M::Tuple) where{N, Ta, D}
    if length(M.parameters) != D
        throw(DimensionMismatch("Expecting the length of M to be $D, but got $(length(M.parameters)) instead"))
    end
    for (y, n) in zip(M.parameters, N.parameters)
        if !(y<:AbstractMatrix)
            throw(ArgumentError("All elements of `M` must be matrices"))
        end
        if last(size(y)) != n
            throw(DimensionMismatch("Incompatible transform dimension"))
        end
    end
    if D > 1
        new_size = tuple((first(size(m)) for m in M.parameters)...)
        tz = tuple((nothing for _ in last(axes(A)))...)
        expr_1 = [:( static_tensor_transform($(Meta.parse("A[" * ":,"^(D-1) * "$k]")), M_red) ) for k in last(axes(A))]
        expr_2 = [:( reshape(B[$k], $(Size(prod(Base.front(new_size))))) ) for k in eachindex(tz)]
        code =
            quote
                M_red = Base.front(M)
                B = tuple($(expr_1...))
                C = tuple($(expr_2...))
                return reshape(hcat(C...) * permutedims(M[end]), $(Size(new_size...)))
            end
        return code
    elseif D == 1
        return :( M[end] * A )
    else
        return :( return A )
    end
end