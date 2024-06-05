module IPM

using LinearAlgebra

using ..Programs

export solve

function solve_dense(prob::DenseSDP; verbose::Bool=false)::Float64
    if verbose
        println("Solving a $(prob.N)-dimensional SDP")
    end
    N,L = prob.N,prob.L
    x = zeros(L)
    # Phase 1
    for step in 1:2000
        M = psd(prob, x)
        F = eigen(M)
        low, idx = findmin(F.values)
        if low > 1e-10
            break
        end
        v = F.vectors[:,idx]
        # This is the vector that minimizes v M v. Get the gradient of v M v.
        dx = zeros(L)
        for i in 1:L
            dx[i] = real(v'prob.M[:,:,i]'v)
        end
        x += 1e-2 * dx
    end
    M = psd(prob, x)
    feasible = minimum(eigvals(M)) > 0
    if !feasible
        error("No (strictly) feasible point found")
    end
    if verbose
        println("Feasible point found.")
    end
 
    # Optimization
    μ = 1.5
    ϵ = 1e-6
    t₀ = 1.

    t = t₀
    while t < 1/ϵ
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
    end
    return prob.H'x + prob.h
end

function solve(sdp::SemidefiniteProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    y = Programs.initial(sdp)
    N = length(y)
    g = zero(y)
    if verbose
        println("Solving SDP: $(N) degrees of freedom")
    end

    # Phase 1
    feasible = false
    for step in 1:100
        badness = Programs.badness!(g, sdp, y)
        if badness ≤ 0
            feasible = true
            break
        end
        for n in 1:N
            y[n] += 1e-2 * g[n]
        end
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
        # Center.
        for step in 1:100
        end
        if verbose
        end
        t = μ*t
    end

    return Programs.objective!(g, sdp, y), y
end

end
