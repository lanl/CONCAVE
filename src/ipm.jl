module IPM

using LinearAlgebra

using ..UnconstrainedOptimization
using ..Programs

export solve

function solve(sdp::SemidefiniteProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = Programs.initial(sdp)
    N = length(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1
    badness = optimize!(y) do ∇,x
        r, g = Programs.badness(sdp,x)
        ∇ .= g
        return r
    end
    if badness > 0
        error("No (strictly) feasible point found")
    end
    if verbose
        println("Feasible point found.")
    end

    # Phase 2
    μ = 1.5
    ϵ = 1e-6
    t₀ = 1.

    t = t₀
    while t < 1/ϵ
        # Center.
        r = optimize!(y) do ∇,x
            barrier, ∇barrier = Programs.barrier(sdp, x)
            obj, ∇obj = Programs.objective(sdp, x)
            ∇ .= ∇obj + ∇barrier/t
            return obj + barrier/t
        end
        if verbose
            println("$t $r")
        end
        t = μ*t
    end

    return Programs.objective(sdp, y)[1], y
end

end
