module CONCAVE

export ConvexProgram, SemidefiniteProgram
export CompositeSDP

export SemidefiniteModel
export primal, dual

export Operator
export Majorana, MajoranaAlgebra, MajoranaOperator
export Pauli, PauliAlgebra, PauliOperator
export Fermion, FermionAlgebra, FermionOperator
export Boson, BosonAlgebra, BosonOperator
export Spins, SpinAlgebra, SpinOperator
export Wick, WickAlgebra, WickOperator

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
