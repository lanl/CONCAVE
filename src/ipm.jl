module IPM

using LinearAlgebra

import Base: size

using ..Programs
using ..UnconstrainedOptimization

import ..Programs: initial, constraints!, objective!

export solve

function feasible(p, y)::Bool
    ok = true
    constraints!(p, y) do f,g,h
        if f isa Matrix
            if minimum(eigvals(Hermitian(f))) < 0
                ok = false
            end
        elseif f isa Real
            if f < 0
                ok = false
            end
        else
            throw(ArgumentError("Expected Real or Matrix"))
        end
    end
    return ok
end

function barrier!(g, h, p, y::Vector{Float64})::Float64
    N = length(y)
    r::Float64 = 0.
    if !isnothing(g)
        g .= 0.
    end
    if !isnothing(h)
        h .= 0.
    end

    function cb(M::Matrix, D, H)
        if !isnothing(H) && H != 0
            error("Hessian is not 0")
        end
        F = eigen(Hermitian(M))
        if minimum(F.values) ≤ 0
            r = Inf
        end
        if r < Inf
            # We use the trace of the logarithm.
            r += -sum(log.(F.values))
            Minv = inv(F)
            if !isnothing(g)
                for n in 1:N
                    g[n] -= real(tr(Minv * D[:,:,n]))
                end
            end
            if !isnothing(h)
                K = size(Minv)[1]
                if true
                    mat′ = similar(Minv)
                    mat = similar(Minv)
                    for n in 1:N
                        mul!(mat′, (@view D[:,:,n]), Minv)
                        mul!(mat, Minv, mat′)
                        # Here mat == Minv * D[:,:,n] * Minv
                        for m in 1:N
                            for i in 1:K, j in 1:K
                                @inbounds h[n,m] += real(mat[i,j] * D[j,i,m])
                            end
                        end
                    end
                end
                if false
                    matn = similar(Minv)
                    matm = similar(Minv)
                    al = @allocations for n in 1:N, m in 1:N
                        mul!(matn, Minv, D[:,:,n])
                        mul!(matm, Minv, D[:,:,m])
                        for i in 1:K
                            for j in 1:K
                                h[n,m] += real(matn[i,j] * matm[j,i])
                            end
                        end
                        #h[n,m] += real(tr(Minv * D[:,:,n] * Minv * D[:,:,m]))
                    end
                    println("$al allocations")
                end
                if false
                    for n in 1:N, m in 1:N
                        h[n,m] += real(tr(Minv * D[:,:,n] * Minv * D[:,:,m]))
                    end
                end
            end
        end
    end

    function cb(f::Real, d, H)
        if !isnothing(H) && H != 0
            error("Hessian is not 0")
        end
        if f ≤ 0
            r = Inf
        end
        if r < Inf
            r += -log(f)
            if !isnothing(g)
                for n in 1:N
                    g[n] -= d[n]/f
                end
            end
            if !isnothing(h)
                for n in 1:N, m in 1:N
                    h[n,m] += d[n] * d[m] / f^2
                end
            end
        end
    end

    constraints!(cb, p, y)
    return r
end

struct Phase1 <: ConvexProgram
    cp::ConvexProgram
end

function size(p::Phase1)
    return 1 + size(p.cp)
end

function initial(p::Phase1)::Vector{Float64}
    y′ = initial(p.cp)
    y = zeros(Float64, 1+length(y′))
    y[2:end] .= y′
    constraints!(p.cp, y′) do f,_grad,_hess
        if f isa Real
            if y[1] + f < 0
                y[1] = -f + 1.0
            end
        elseif f isa Matrix
            f = minimum(eigvals(Hermitian(f)))
            if y[1] + f < 0
                y[1] = -f + 1.0
            end
        else
            throw(ArgumentError("Expected Real or Matrix"))
        end
    end
    return y
end

function objective!(g, p::Phase1, y::Vector{Float64})::Float64
    g[1] = 1.
    g[2:end] .= 0.
    return y[1]
end

function constraints!(cb, p::Phase1, y::Vector{Float64})
    N = length(y)-1
    s = y[1]
    function fn(M::Matrix, D, H)
        @assert H == 0
        # TODO should pass a matrix into cb(), really.
        F = eigen(Hermitian(M))
        f = F.values[1]
        v = F.vectors[:,1]
        g′ = zeros(Float64, length(D)+1)
        g′[1] = 1.
        for n in 1:N
            g′[1+n] = real(v' * D[:,:,n] * v)
        end
        cb(s+f, g′)
    end

    function fn(f::Real, d, h)
        @assert h == 0
        g = zeros(Float64, length(d)+1)
        g[1] = 1.
        for n in 1:N
            g[1+n] = d[n]
        end
        cb(s+f, g)
    end

    constraints!(fn, p.cp, y[2:end])
end

function feasible_initial_new(prog::ConvexProgram; verbose::Bool=false)::Vector{Float64}
    if verbose
        println(stderr, "Finding feasible initial point...")
    end

    # Construct phase-1 problem.
    p1 = Phase1(prog)
    y = initial(p1)
    # TODO calculate t₀
    solve(p1, y) do obj
        return obj < 0
    end
    y′ = y[2:end]

    # TODO return t₀

    if !feasible(prog, y′)
        error("No (strictly) feasible point found.")
    end

    return y′
end

function feasible_initial(prog::ConvexProgram; verbose::Bool=false)::Vector{Float64}
    if verbose
        println(stderr, "Finding feasible initial point...")
    end

    N = size(prog)
    y = initial(prog)
    g = zero(y)

    minimize!(BFGS, y) do g, y
        r::Float64 = 0.
        if !isnothing(g)
            g .= 0.0
        end
        constraints!(prog, y) do M,D,H
            if any(isinf.(M)) || any(isnan.(M))
                r = Inf
                return
            end
            F = eigen(Hermitian(M))
            f = F.values[1]
            v = F.vectors[:,1]
            if f ≤ 0
                r -= f
                if !isnothing(g)
                    for n in 1:N
                        g[n] -= real(v' * D[:,:,n] * v)
                    end
                end
            end
        end
        return r
    end

    if !feasible(prog, y)
        error("No (strictly) feasible point found.")
    end

    return y
end

# TODO allow t₀ in solver

function solve(term::Function, prog::ConvexProgram, y)
    return solve(prog, y; term=term)
end

function solve(prog::ConvexProgram, y; verbose::Bool=false, term=nothing)::Tuple{Float64, Vector{Float64}}
    if !feasible(prog, y)
        error("Initial point was not (strictly) feasible")
    end

    N = length(y)
    g = zero(y)

    μ = 2
    ϵ = 1e-10
    t₀ = 1.0e-4

    t = t₀
    while t < 1/ϵ
        # Center.
        v = minimize!(Newton, y) do g, h, y
            if any(isnan.(y)) || any(isinf.(y))
                return Inf
            end
            gobj, gbar = zero(g), zero(g)
            obj = objective!(gobj, prog, y)
            bar = barrier!(gbar, h, prog, y)
            r = obj + bar/t
            @. g = gobj + gbar/t
            if !isnothing(h)
                h ./= t
            end
            return r
        end
        obj = objective!(g, prog, y)
        if !isnothing(term)
            if term(obj)
                return obj, y
            end
        end
        if verbose
            println(stderr, t, " ", v, "   ", obj)
        end
        t = μ*t
    end

    return objective!(g, prog, y), y
end

function solve(prog::ConvexProgram; verbose::Bool=false)::Tuple{Float64, Vector{Float64}}
    if verbose
        println(stderr, "Solving $(typeof(prog))")
    end
    y = feasible_initial(prog; verbose=verbose)

    if verbose
        println(stderr, "Performing phase-2 optimization...")
    end
    return solve(prog, y; verbose=verbose)
end

end
