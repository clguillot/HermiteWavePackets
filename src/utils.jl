
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
        init_code = :( λ = zero($T) )
        if eltype(first(X)) <: Complex
            prep_code = quote
                            re_X1 = real.(first(X))
                            im_X1 = imag.(first(X))
                        end
        else
            prep_code = :()
        end
        return_code = :( return λ )
        loop_code = :()
        array_access = Meta.parse("A[" * prod("j$k," for k in eachindex(S)) * "]")
        array_access_drop = Meta.parse("A[" * ":," * prod("j$k," for k in Iterators.drop(eachindex(S), firstindex(S))) * "]")
        for (ax, k, l) in zip(axes(A), 1:length(S), eachindex(X))
            jk_symb = Symbol("j$k")
            xk_symb = Symbol("x$k")
            μk_symb = Symbol("μ$k")
            μkp1_symb = Symbol("μ$(k+1)")
            if k == 1
                if eltype(first(X)) <: Complex
                    if k == length(S)
                        loop_code =
                            :(
                                λ += dot(re_X1, $array_access_drop) + im * dot(im_X1, $array_access_drop)
                            )
                    else
                        loop_code =
                            :(
                                λ += $μkp1_symb * (dot(re_X1, $array_access_drop) + im * dot(im_X1, $array_access_drop))
                            )
                    end
                else
                    if k == length(S)
                        loop_code =
                            :(
                                λ += dot(X[$l], $array_access_drop)
                            )
                    else
                        loop_code =
                            :(
                                λ += $μkp1_symb * dot(X[$l], $array_access_drop)
                            )
                    end
                end
            else
                if k == length(S)
                    μ_code = :( $μk_symb = $xk_symb )
                else
                    μ_code = :( $μk_symb = $μkp1_symb * $xk_symb )
                end
                loop_code =
                    :(
                        for ($jk_symb, $xk_symb) in zip($ax, X[$l])
                            $μ_code
                            $loop_code
                        end
                    )
            end
        end
        return quote
            $prep_code
            $init_code
            $loop_code
            $return_code
        end
    else
        throw(ErrorException("Not implemented..."))
    end
end