module Programs

export ConvexProgram, SemidefiniteProgram, DenseSDP
export dual

abstract type ConvexProgram end

abstract type SemidefiniteProgram <: ConvexProgram end

struct DenseSDP <: SemidefiniteProgram
end

function dual(primal::DenseSDP)::DenseSDP
    # TODO
end

end
