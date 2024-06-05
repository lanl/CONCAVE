module Programs

export ConvexProgram, SemidefiniteProgram
export CompositeSDP, DenseSDP
export dual

using LinearAlgebra

abstract type ConvexProgram end

abstract type SemidefiniteProgram <: ConvexProgram end

struct DenseSDP <: SemidefiniteProgram
    A::Array{ComplexF64,3}
    b::Vector{Float64}
    C::Matrix{ComplexF64}
end

function dual(primal::DenseSDP)::DenseSDP
    # TODO
end

struct CompositeSDP <: SemidefiniteProgram
    M::Vector{Array{ComplexF64,3}}
    h::Vector{Float64}

    function CompositeSDP(N::Int, D::Vector{Int})
        h = zeros(Float64, N)
        M = Vector{Array{ComplexF64,3}}()
        for d in D
            push!(M, zeros(ComplexF64, (d,d,N)))
        end
        return new(M,h)
    end
end

function dual(primal::CompositeSDP)::CompositeSDP
end

function initial(sdp::CompositeSDP)::Vector{Float64}
    return zeros(Float64, length(sdp.h))
end

function badness!(∇::AbstractVector{Float64}, sdp::CompositeSDP, y::AbstractVector{Float64})::Float64
    return 0.
end

function barrier!(∇::AbstractVector{Float64}, sdp::CompositeSDP, y::AbstractVector{Float64})::Float64
    return 0.
end

function objective!(∇::AbstractVector{Float64}, sdp::CompositeSDP, y::AbstractVector{Float64})::Float64
    return sdp.h ⋅ y
end

end
