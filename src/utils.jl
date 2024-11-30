
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

@generated function fitting_float(x::T) where{T<:Real}
    Tf = fitting_float(T)
    return :( $Tf )
end

@generated function fitting_float(x::AbstractArray{T}) where{T<:Number}
    Tf = fitting_float(T)
    return :( $Tf )
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