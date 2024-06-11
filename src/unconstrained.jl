module UnconstrainedOptimization

export optimize!

using LinearAlgebra

struct BacktrackingLineSearch
    α₀::Float64
    τ::Float64
    c::Float64
end

function BacktrackingLineSearch()
    return BacktrackingLineSearch(1, 0.5, 0.5)
end

function optimize!(f!, y::Vector{Float64})
    bls = BacktrackingLineSearch()
    ∇ = zero(y)
    ∇′ = zero(y)
    y′ = zero(y)
    # TODO when to terminate gradient descent?
    for step in 1:100
        α = bls.α₀
        r = f!(∇, y)
        m = - ∇ ⋅ ∇
        t = - bls.c * m
        y′ = y - α * ∇
        r′ = f!(∇′, y′)
        while r - r′ < α * t
            α *= bls.τ
            y′ = y - α * ∇
            r′ = f!(∇′, y′)
        end
        y .= y′
    end
    return f!(∇, y)[1]
end

end
