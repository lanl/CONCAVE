module Programs

export ConvexProgram, SemidefiniteProgram
export CompositeSDP

export SemidefiniteModel
export primal, dual

import Base: push!

using LinearAlgebra

abstract type ConvexProgram end

abstract type SemidefiniteProgram <: ConvexProgram end

struct CompositeSDP <: SemidefiniteProgram
    D::Vector{Int}
    M₀::Vector{Matrix{ComplexF64}}
    M::Vector{Array{ComplexF64,3}}
    h::Vector{Float64}

    function CompositeSDP(N::Int, D::Vector{Int})
        h = zeros(Float64, N)
        M = Vector{Array{ComplexF64,3}}()
        M₀ = Vector{Matrix{ComplexF64}}()
        for d in D
            push!(M, zeros(ComplexF64, (d,d,N)))
            push!(M₀, zeros(ComplexF64, (d,d)))
        end
        return new(copy(D),M₀,M,h)
    end
end

function initial(sdp::CompositeSDP)::Vector{Float64}
    return randn(Float64, length(sdp.h))
end

function badness(sdp::CompositeSDP, y::AbstractVector{Float64})::Tuple{Float64, Vector{Float64}}
    N = length(sdp.h)
    ∇ = zero(sdp.h)
    r = 0.
    for (i,d) in enumerate(sdp.D)
        A = copy(sdp.M₀[i])
        for n in 1:N
            A += y[n] * sdp.M[i][:,:,n]
        end
        F = eigen(Hermitian(A))
        for k in 1:d
            if F.values[k] ≤ 0
                r += -F.values[k]
                v = F.vectors[:,k]
                for n in 1:N
                    ∇[n] += -real(v'sdp.M[i][:,:,n]'v)
                end
            end
        end
    end
    return r, ∇
end

function barrier(sdp::CompositeSDP, y::AbstractVector{Float64})::Tuple{Float64, Vector{Float64}}
    N = length(sdp.h)
    ∇ = zero(sdp.h)
    r = 0.
    for (i,d) in enumerate(sdp.D)
        A = copy(sdp.M₀[i])
        for n in 1:N
            A += y[n] * sdp.M[i][:,:,n]
        end
        d = det(Hermitian(A))
        if d ≤ 0
            return Inf, ∇
        end
        r -= real(log(d))
        Ainv = inv(A)
        for n in 1:N
            ∇[n] -= real(tr(Ainv * sdp.M[i][:,:,n]))
        end
    end
    return r, ∇
end

function objective(sdp::CompositeSDP, y::AbstractVector{Float64})::Tuple{Float64, Vector{Float64}}
    return sdp.h ⋅ y, sdp.h
end

struct SemidefiniteModel{T}
    variables::Vector{T}

    function SemidefiniteModel{T}() where {T}
        variables = Vector{T}()
        new{T}(variables)
    end
end

struct Variable{T}
    name::T
end

struct Equality{T}
end

struct LinearExpression{T}
end

function (push!)(m::SemidefiniteModel{T}, var::Variable{T}) where {T}
    push!(m.variables, var.name)
    return m
end

function primal(m::SemidefiniteModel)::SemidefiniteProgram
end

function dual(m::SemidefiniteModel)::SemidefiniteProgram
end

end
