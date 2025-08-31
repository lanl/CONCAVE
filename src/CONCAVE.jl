module CONCAVE

export ConvexProgram, SemidefiniteProgram
export CompositeSDP

export SemidefiniteModel
export primal, dual

export @algebra

export add!, scale!

include("programs.jl")
include("algebra.jl")
include("unconstrained.jl")
include("ipm.jl")
include("hamiltonians.jl")
include("splines.jl")
include("utilities.jl")

using .Programs
using .Algebras
using .IPM

end
