struct NullNumber end  # Struct to represent the fully absorbing zero number

# Define arithmetic properties
# Multiplication
Base.:*(::NullNumber, ::NullNumber) = NullNumber()
Base.:*(::Number, ::NullNumber) = NullNumber()
Base.:*(::NullNumber, ::Number) = NullNumber()
# Addition
Base.:+(::NullNumber, ::NullNumber) = NullNumber()
Base.:+(x::Number, ::NullNumber) = x
Base.:+(::NullNumber, x::Number) = x
# Substraction
Base.:-(::NullNumber, ::NullNumber) = NullNumber()
Base.:-(x::Number, ::NullNumber) = x
Base.:-(::NullNumber, x::Number) = -x
Base.:-(::NullNumber) = NullNumber()
# Division
Base.:/(::NullNumber, ::NullNumber) = throw(DivideError())
Base.:/(::NullNumber, ::Number) = NullNumber()
# Power
Base.:^(::NullNumber, x::Number) = NullNumber()
# Conjugate
Base.conj(::NullNumber) = NullNumber()
# Absolute value
Base.abs(::NullNumber) = NullNumber()
Base.abs2(::NullNumber) = NullNumber()

# Special imag
imagz(::Real) = NullNumber()
imagz(z::Number) = imag(z)

# Defines special operations with SArray
Base.:*(::NullNumber, x::AbstractArray{<:Union{Number, NullNumber}}) = similar(x, NullNumber)
#
Base.:\(::SDiagonal{D, <:Number}, ::SVector{D, NullNumber}) where D = zeros(SVector{D, NullNumber})
Base.:\(::UpperTriangular{<:Number, <:SMatrix{Dx, Dy}}, ::SVector{Dy, NullNumber}) where{Dx, Dy} = zeros(SVector{Dx, NullNumber})
Base.:\(::LowerTriangular{<:Number, <:SMatrix{Dx, Dy}}, ::SVector{Dy, NullNumber}) where{Dx, Dy} = zeros(SVector{Dx, NullNumber})
LinearAlgebra.dot(::SArray{N, NullNumber}, ::SArray{N, <:Union{Number, NullNumber}}) where N = NullNumber()
LinearAlgebra.dot(::SArray{N, <:Number}, ::SArray{N, NullNumber}) where N = NullNumber()
LinearAlgebra.dot(::NullNumber, ::Union{Number, NullNumber}) = NullNumber()
LinearAlgebra.dot(::Number, ::NullNumber) = NullNumber()

# Special functions
Base.exp(::NullNumber) = true
Base.cis(::NullNumber) = true

# Define promotion rules
LinearAlgebra.symmetric_type(::Type{NullNumber}) = NullNumber
LinearAlgebra.symmetric(A::NullNumber, uplo::Symbol=:U) = NullNumber()
Base.promote_rule(::Type{NullNumber}, ::Type{T}) where{T<:Number} = T
Base.promote_rule(::Type{T}, ::Type{NullNumber}) where{T<:Number} = T

# 
Base.zero(::Type{NullNumber}) = NullNumber()
Base.zero(::NullNumber) = NullNumber()
Base.convert(::Type{T}, ::NullNumber) where{T<:Number} = zero(T)

# Iterate
Base.iterate(::NullNumber) = (NullNumber(), nothing)
Base.iterate(::NullNumber, ::Any) = nothing

# Define printing
Base.show(io::IO, ::NullNumber) = print(io, "NullNumber()")
