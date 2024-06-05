module CONCAVE

export ConvexProgram, SemidefiniteProgram, DenseSDP
export dual

export Operator
export Majorana, MajoranaAlgebra
export Pauli, PauliAlgebra
export Fermion, FermionAlgebra
export Boson, BosonAlgebra
export Wick, WickAlgebra

include("programs.jl")
include("algebra.jl")
include("ipm.jl")

using .Programs
using .Algebras
using .IPM

end
