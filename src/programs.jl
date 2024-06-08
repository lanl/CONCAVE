module Programs

export ConvexProgram, SemidefiniteProgram
export CompositeSDP
export dual

using LinearAlgebra

abstract type ConvexProgram end

abstract type SemidefiniteProgram <: ConvexProgram end

struct CompositeSDP <: SemidefiniteProgram
    D::Vector{Int}
    M::Vector{Array{ComplexF64,3}}
    h::Vector{Float64}

    function CompositeSDP(N::Int, D::Vector{Int})
        h = zeros(Float64, N)
        M = Vector{Array{ComplexF64,3}}()
        for d in D
            push!(M, zeros(ComplexF64, (d,d,N)))
        end
        return new(copy(D),M,h)
    end
end

function dual(primal::CompositeSDP)::CompositeSDP
    # TODO
end

function initial(sdp::CompositeSDP)::Vector{Float64}
    return zeros(Float64, length(sdp.h))
end

function badness(sdp::CompositeSDP, y::AbstractVector{Float64})::Tuple{Float64, Vector{Float64}}
    N = length(sdp.h)
    ∇ = zero(sdp.h)
    r = 0.
    for (i,d) in enumerate(sdp.D)
        A = zeros(ComplexF64, (d,d))
        for n in 1:N
            A += y[n] * sdp.M[i][:,:,n]
        end
        F = eigen(Hermitian(A))
        for k in 1:d
            if F.values[k] < 0
                r += -F.values[k]
                for n in 1:N
                    ∇ += real(v'sdp.M[:,:,n]'v)
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
        A = zeros(ComplexF64, (d,d))
        for n in 1:N
            A += y[n] * sdp.M[i][:,:,n]
        end
        r += real(logdet(Hermitian(A)))
        Ainv = inv(A)
        for n in 1:N
            ∇ += real(tr(Ainv * sdp.M[i][:,:,n]))
        end
    end
    return r, ∇
end

function objective(sdp::CompositeSDP, y::AbstractVector{Float64})::Tuple{Float64, Vector{Float64}}
    return sdp.h ⋅ y, y
end

end
