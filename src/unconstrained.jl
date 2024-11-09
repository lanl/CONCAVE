# Unconstrained optimization routines for convex functions.

module UnconstrainedOptimization

export GradientDescent, LineSearch, BFGS, Newton
export minimize!

using LinearAlgebra

# Naive gradient descent
struct GradientDescent
end

function (gd::GradientDescent)(f!, y::Vector{Float64})::Float64
    N = length(y)
    δ = 1e-1
    ∇::Vector{Float64} = zero(y)
    ∇′::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    for step in 1:500
        r₀ = f!(∇′, y)
        for k in 1:100
            r = f!(∇, y)
            y′ .= y
            for n in 1:N
                y′[n] -= δ * ∇[n]
            end
            r′ = f!(∇′, y′)
            if r′ < r
                y .= y′
            end
        end
        r = f!(∇′, y)
        if r ≥ r₀
            if δ < 1e-6
                return r
            else
                δ /= 2.
            end
        end
    end
    return f!(∇, y)[1]
end

# Gradient descent with line search
struct LineSearch
    α₀::Float64
    ϵ::Float64
end

function LineSearch()
    return LineSearch(1, 1e-6)
end

function (gd::LineSearch)(f!, y::Vector{Float64})::Float64
    N = length(y)
    ∇::Vector{Float64} = zero(y)
    ∇′::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)
    α = 1.
    αmin = 1e-10
    δmin = 1e-10
    for step in 1:1000000
        r₀ = f!(∇, y)
        if step%200000 == 0
            println("   Taking long! step=$step   α=$α    r₀=$r₀")
            println(y)
        end
        function at!(α::Float64)::Float64
            for n in 1:N
                y′[n] = y[n] - α * ∇[n]
            end
            return f!(∇′, y′)
        end
        r = at!(α)
        while r > r₀ && α > αmin
            α /= 2
            r = at!(α)
        end

        while α > αmin
            r′ = at!(α/2)
            if r′ < r
                r = r′
                α /= 2
            else
                break
            end
        end
        δ = r₀ - r
        if α < αmin || δ < δmin
            break
        end
        y′ = y - α * ∇
        y .= y′
        α *= 10
    end
    return f!(∇, y)[1]
end

struct BFGS
end

function (bfgs::BFGS)(f!, y::Vector{Float64}; H0=nothing)::Float64
    N = length(y)
    ∇::Vector{Float64} = zero(y)
    ∇′::Vector{Float64} = zero(y)
    v::Vector{Float64} = zero(y)
    d::Vector{Float64} = zero(y)
    y′::Vector{Float64} = zero(y)

    α = 1.
    αmin = 1e-12
    δmin = 1e-12

    # Initial guess of inverse Hessian (just guess the identity).
    H = zeros(Float64, (N,N))
    if isnothing(H0)
        for n in 1:N
            H[n,n] = 1.0
        end
    else
        H .= H0
    end

    nsteps = 0
    for step in 1:100000
        nsteps += 1
        # Get initial value and gradient.
        r₀ = f!(∇, y)
        if isnan(r₀)
            error("Reached NaN")
        end
        #println("step=$step   $r₀")
        g = H * ∇
        g ./= norm(g)

        # Line search
        function at!(α::Float64)::Float64
            for n in 1:N
                y′[n] = y[n] - α * g[n]
            end
            return f!(nothing, y′)
        end
        r = at!(α)
        while r > r₀ && α > αmin
            α /= 2
            r = at!(α)
        end

        while α > αmin
            r′ = at!(α/2)
            if r′ < r
                r = r′
                α /= 2
            else
                break
            end
        end
        δ = r₀ - r
        if α < αmin || δ < δmin
            break
        end
        v .= -α*g
        y .+= v
        α *= 10

        # Update the gradient
        f!(∇′, y)
        #println(f!(∇′, y), "   ", y)
        #println("     ", ∇′)
        #println(H[1,1])
        #println()

        # Update inverse Hessian.
        d = ∇′ - ∇
        den = v' * d
        den += 1e-8  # Crude damping
        H = H + (v' * d + d' * H * d) * (v * v') / den^2 - (H * d * v' + v * d' * H) / den
        if rand() < 1 && false
            F = eigen(Hermitian(H))
            println("   DUMPING INVERSE-HESSIAN EIGENDECOMPOSITION")
            println("     VALUES: ", F.values)
            println("     LOWEST VECTOR: ", F.vectors[:,1])
            println("     HIGHEST VECTOR: ", F.vectors[:,end])
            println("  ", y)
        end
    end

    if !isnothing(H0)
        H0 .= H
    end

    #printstyled(stderr, "BFGS took $nsteps steps\n", color=:light_black)

    return f!(nothing, y)[1]
end

struct LBFGS
end

struct Newton
end

function (newton::Newton)(f!, y::Vector{Float64})::Float64
    N = length(y)
    y′ = zero(y)
    g = zeros(Float64, N)
    h = zeros(Float64, (N,N))
    v::Float64 = f!(g, h, y)
    nsteps = 0
    while true
        if false
            F = eigen(Symmetric(h))
            F.values .+= 1e-13 * maximum(F.values)
            F.values .+= 1e-50
            hinv = inv(F)
            dy = -hinv * g
        end
        dy = -h \ g

        # Termination
        #δ = -(g' * hinv * g) / 4
        δ = (g ⋅ dy) / 4
        if norm(dy) < 1e-10 || abs(δ) / abs(v) < 1e-10
            break
        end

        # Backtracking line search
        α = 1.0
        m = g ⋅ dy
        @. y′ = y + α * dy
        v′ = f!(g, h, y′)
        while v′ > v + 0.5 * α * m && α > 1e-30
            α *= 0.5
            @. y′ = y + α * dy
            v′ = f!(g, h, y′)
        end
        v = v′
        @. y = y′

        if α < 1e-30
            break
        end

        nsteps += 1
    end
    return v
end

function minimize!(f!, alg, y::Vector{Float64}; kwargs...)::Float64
    return alg()(f!, y; kwargs...)
end

end
