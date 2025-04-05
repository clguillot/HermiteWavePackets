struct NullNumber end  # Struct to represent the fully absorbing zero number

# Define arithmetic properties
# Multiplication
@inline Base.:*(::NullNumber, ::NullNumber) = NullNumber()
@inline Base.:*(::Number, ::NullNumber) = NullNumber()
@inline Base.:*(::NullNumber, ::Number) = NullNumber()
# Addition
@inline Base.:+(::NullNumber, ::NullNumber) = NullNumber()
@inline Base.:+(x::Number, ::NullNumber) = x
@inline Base.:+(::NullNumber, x::Number) = x
# Substraction
@inline Base.:-(::NullNumber, ::NullNumber) = NullNumber()
@inline Base.:-(x::Number, ::NullNumber) = x
@inline Base.:-(::NullNumber, x::Number) = -x
@inline Base.:-(::NullNumber) = NullNumber()
# Division
@inline Base.:/(::NullNumber, ::NullNumber) = throw(DivideError())
@inline Base.:/(::NullNumber, ::Number) = NullNumber()

# Defines special operations with SArray
@inline Base.:*(::NullNumber, ::SArray{N, <:Union{Number, NullNumber}}) where N = zeros(SArray{N, NullNumber})
@inline LinearAlgebra.dot(::SArray{N, NullNumber}, ::SArray{N, <:Union{Number, NullNumber}}) where N = NullNumber()
@inline LinearAlgebra.dot(::SArray{N, <:Number}, ::SArray{N, NullNumber}) where N = NullNumber()

# Special functions
@inline Base.exp(::NullNumber) = true
@inline Base.cis(::NullNumber) = true

# Define promotion rules
@inline Base.promote_rule(::Type{NullNumber}, ::Type{T}) where{T<:Number} = T
@inline Base.promote_rule(::Type{T}, ::Type{NullNumber}) where{T<:Number} = T

# 
@inline Base.zero(::Type{NullNumber}) = NullNumber()
@inline Base.zero(::NullNumber) = NullNumber()
@inline Base.convert(::Type{T}, ::NullNumber) where{T<:Number} = zero(T)

# Define printing
@inline Base.show(io::IO, ::NullNumber) = print(io, "NullNumber()")
