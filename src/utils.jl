
@generated function fitting_float(::Type{T}) where{T<:Number}
    prec = precision(real(T))
    if prec == precision(Float16)
        return :( Float16 )
    elseif prec == precision(Float32)
        return :( Float32 )
    elseif prec == precision(Float64)
        return :( Float64 )
    else
        throw(ArgumentError("The precision format of $T is unsupported"))
    end
end

function fitting_float(x::T) where{T<:Real}
    return fitting_float(T)
end

function fitting_float(x::AbstractArray{T}) where{T<:Number}
    return fitting_float(T)
end

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
        α₀ = α0
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

    STATIC TENSOR CONTRACTION

=#
@generated function static_tensor_contraction(A::SArray{N, <:Number}, X::SVector...) where N
    S = N.parameters
    for (k, s, x) in zip(eachindex(S), S, X)
        if length(x) != s
            throw(DimensionMismatch("DimensionMismatch: Incompatible contraction length for $k-th index"))
        end
        if !(eltype(x)<:Number)
            throw(ArgumentError("ArgumentError: Expecting element type of all vectors to be a subtype of $Number, but argument type for the $k-th contraction vector has element type $(eltype(x))"))
        end
    end
    T = promote_type(eltype(A), (eltype.(X))...)
    if length(X) == 0
        return A
    elseif length(X) > length(A)
        throw(ArgumentError("ArgumentError: The number of vectors for contraction exceeds the available indices in A"))
    elseif length(S) == length(X)
        if !all(n -> n > 0, size(A))
            return :( return zero($T) )
        else
            if eltype(first(X)) <: Complex
                prep_code = quote
                                re_X1 = real.(first(X))
                                im_X1 = imag.(first(X))
                            end
            else
                prep_code = :()
            end
            loop_code = :()
            array_access = Meta.parse("A[" * prod("j$k," for k in eachindex(S)) * "]")
            array_access_drop = Meta.parse("A[" * ":," * prod("j$k," for k in Iterators.drop(eachindex(S), firstindex(S))) * "]")
            for (ax, k, l) in zip(axes(A), 1:length(S), eachindex(X))
                jk = Symbol("j$k")
                xk = Symbol("x$k")
                φk = Symbol("φ$k")
                φkm1 = Symbol("φ$(k-1)")
                if k == 1
                    if eltype(first(X)) <: Complex
                        loop_code =
                            :(
                                $φk = dot(re_X1, $array_access_drop) + im * dot(im_X1, $array_access_drop)
                            )
                    else
                        loop_code =
                            :(
                                $φk = dot(X[$l], $array_access_drop)
                            )
                    end
                else
                    loop_code =
                        quote
                            $φk = zero($T)
                            for ($jk, $xk) in zip($ax, X[$l])
                                $loop_code
                                $φk += $xk * $φkm1
                            end
                        end
                end
            end
            return quote
                $prep_code
                $loop_code
                return $(Symbol("φ$(length(S))"))
            end
        end
    else
        throw(ErrorException("Not implemented..."))
    end
end