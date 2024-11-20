module ComplexHermiteFct

# Custom mathematical operations compatible with autodifferentiation
include("math.jl")

include("real/gaussian1d.jl")
include("real/hermite_fct1d.jl")
include("complex/complex_gaussian1d.jl")
include("complex/complex_hermite_fct1d.jl")

end
