module IPM

using LinearAlgebra

using ..Programs

export solve

#=
        h = 1e-2
        # Center
        for step in 1:800
            M = psd(prob, x)
            obj = prob.H'x - real(logdet(M))/t

            # Barrier gradient
            db = zeros(L)
            Minv = inv(M)
            for i in 1:L
                db[i] = -real(tr(Minv*prob.M[:,:,i]))
            end

            # Step
            d = prob.H + db/t
            x_ = x - h*d
            M_ = psd(prob, x_)
            obj_ = prob.H'x_ - real(logdet(M_))/t
            feasible = minimum(eigvals(M_)) > 0
            if !feasible || obj_ > obj
                h = h/2
            else
                x = x_
            end
        end
        if verbose
            println("$t $(prob.H'x+prob.h) $(minimum(eigvals(psd(prob,x))))")
        end
        t = μ*t
=#

function solve(sdp::SemidefiniteProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = Programs.initial(sdp)
    N = length(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1
    feasible = false
    for step in 1:100
        badness, ∇badness = Programs.badness(sdp, y)
        if badness ≤ 0
            feasible = true
            break
        end
        y += 1e-2 * ∇badness[n]
    end
    if !feasible
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
        h = 1e-2
        # Center.
        for step in 1:100
            barrier, ∇barrier = Programs.barrier(sdp, y)
            obj, ∇obj = Programs.objective(g, sdp, y)
            y -= h * (∇obj + ∇barrier / t)
            # TODO backtrack
        end
        if verbose
            println("$t $obj")
        end
        t = μ*t
    end

    return Programs.objective!(g, sdp, y), y
end

end
