module CONCAVE

export ConvexProgram, SemidefiniteProgram, DenseSDP

export Operator
export Majorana, MajoranaAlgebra

include("programs.jl")
include("algebra.jl")
include("ipm.jl")

using .Programs
using .Algebras
using .IPM

end
