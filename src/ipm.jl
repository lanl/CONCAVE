module IPM

using LinearAlgebra

import Base: size

using ..Programs
using ..UnconstrainedOptimization

import ..Programs: initial, constraints!, objective!

export solve

function feasible(p, y)::Bool
    ok = true
    constraints!(p, y) do f,g
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

function barrier!(g, p, y::Vector{Float64})::Float64
    N = length(g)
    r::Float64 = 0.
    g .= 0.

    function cb(M::Matrix, D)
        F = eigen(Hermitian(M))
        if minimum(F.values) ≤ 0
            r = Inf
        end
        if r < Inf
            # We just use the minimum eigenvalue.
            f = F.values[1]
            r += -log(f)
            v = F.vectors[:,1]
            for n in 1:N
                g[n] -= real(v' * D[:,:,n] * v)/f
            end
        end
    end

    function cb(f::Real, d)
        if f ≤ 0
            r = Inf
        end
        if r < Inf
            r += -log(f)
            for n in 1:N
                g[n] -= d[n]/f
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
    constraints!(p.cp, y′) do f,g
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
    function fn(M::Matrix, D)
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

    function fn(f::Real, d)
        g = zeros(Float64, length(d)+1)
        g[1] = 1.
        for n in 1:N
            g[1+n] = d[n]
        end
        cb(s+f, g)
    end

    constraints!(fn, p.cp, y[2:end])
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
        g .= 0.0
        constraints!(prog, y) do M,D
            if any(isinf.(M)) || any(isnan.(M))
                r = Inf
                return
            end
            F = eigen(Hermitian(M))
            f = F.values[1]
            v = F.vectors[:,1]
            if f ≤ 0
                r -= f
                for n in 1:N
                    g[n] -= real(v' * D[:,:,n] * v)
                end
            end
        end
        return r
    end

    if !feasible(prog, y)
        error("No feasible point found.")
    end

    return y
end

function solve(prog::ConvexProgram, y; verbose::Bool=false, gd=BFGS, early=nothing)::Tuple{Float64, Vector{Float64}}
    if !feasible(prog, y)
        error("Initial point was not feasible")
    end

    N = length(y)
    g = zero(y)

    μ = 1.1
    ϵ = 1e-10
    t₀ = 1.0e-3

    t = t₀
    while t < 1/ϵ
        # Center.
        v = minimize!(gd, y) do g, y
            if any(isnan.(y)) || any(isinf.(y))
                return Inf
            end
            gobj, gbar = zero(g), zero(g)
            obj = objective!(gobj, prog, y)
            bar = barrier!(gbar, prog, y)
            r = obj + bar/t
            for n in 1:N
                g[n] = gobj[n] + gbar[n]/t
            end
            return r
        end
        obj = objective!(g, prog, y)
        if verbose
            println(stderr, t, " ", v, "   ", obj)
        end
        if !isnothing(early)
            if early(y)
                break
            end
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
