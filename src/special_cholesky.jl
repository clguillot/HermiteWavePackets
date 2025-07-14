#=
    Computes the analytic extension of the Cholesky decomposition (LᵀL=A)
    Returns the UpperTriangular part of the decomposition
    Adapted from github.com/JuliaArrays/StaticArrays.jl
=#

# Diagonal
function special_cholesky(D::SDiagonal)
    any(x -> real(x) < 0, D.diag) && throw(LinearAlgebra.PosDefException(1))
    C = sqrt.(D.diag)
    return SDiagonal(C)
end

# Dense
function special_cholesky(A::Symmetric{<:Number, <:StaticMatrix})
    _special_cholesky(Size(A), A.data)
end

@inline function _sp_chol_failure(info)
    throw(LinearAlgebra.PosDefException(info))
end
# x < zero(x) is check used in `sqrt`, letting LLVM eliminate that check and remove error code.
@inline _sp_nonpdcheck(x::Number) = real(x) ≥ zero(real(x))
@inline _sp_nonpdcheck(x) = x == x

@generated function _special_cholesky(::Size{S}, A::StaticMatrix{M, M}) where{S, M}
    @assert (M,M) == S
    # M > 24 && return :(_cholesky_large(Size{$S}(), A, check))
    q = Expr(:block)
    for n ∈ 1:M
        for m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n = @inbounds A[$n, $m]))
        end
        for k ∈ 1:n-1, m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            L_m_k = Symbol(:L_,m,:_,k)
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_m_n = muladd(-$L_m_k, $L_n_k, $L_m_n)))
        end
        L_n_n = Symbol(:L_,n,:_,n)
        L_n_n_ltz = Symbol(:L_,n,:_,n,:_,:ltz)
        push!(q.args, :(_sp_nonpdcheck($L_n_n) || return _sp_chol_failure($n)))
        push!(q.args, :($L_n_n = sqrt($L_n_n)))
        Linv_n_n = Symbol(:Linv_,n,:_,n)
        push!(q.args, :($Linv_n_n = inv($L_n_n)))
        for m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n *= $Linv_n_n))
        end
    end
    push!(q.args, :(T = eltype(A)))
    ret = Expr(:tuple)
    for n ∈ 1:M
        for m ∈ 1:n
            push!(ret.args, Symbol(:L_,n,:_,m))
        end
        for m ∈ n+1:M
            push!(ret.args, :(zero(T)))
        end
    end
    push!(q.args, :(UpperTriangular(typeof(A)($ret))))
    return Expr(:block, Expr(:meta, :inline), q)
end