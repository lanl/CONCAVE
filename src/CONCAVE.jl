module CONCAVE

export ConvexProgram, SemidefiniteProgram
export CompositeSDP

export SemidefiniteModel
export primal, dual

export Operator
export Majorana, MajoranaAlgebra
export Pauli, PauliAlgebra
export Fermion, FermionAlgebra
export Boson, BosonAlgebra
export Spin, SpinAlgebra
export Wick, WickAlgebra

include("programs.jl")
include("algebra.jl")
include("unconstrained.jl")
include("ipm.jl")
include("hamiltonians.jl")

using .Programs
using .Algebras
using .IPM

end
