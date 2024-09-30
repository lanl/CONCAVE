#=
Outstanding questions

Right now Λ! never uses B. That is, the delta-function at t=0 is just ignored.
Can we get better bounds by not ignoring it?

=#

using ArgParse
using LinearAlgebra: tr

using CONCAVE

import Base: size
import CONCAVE.Programs: initial, constraints!, objective!

demo(s::Symbol) = demo(Val(s))

struct AHOProgram <: ConvexProgram
    T::Float64
    K::Int
    N::Int
    O::Matrix{ComplexF64}
    A::Vector{Matrix{ComplexF64}}
    B::Vector{Matrix{ComplexF64}}
    C::Vector{Matrix{ComplexF64}}
    D::Vector{Matrix{ComplexF64}}
    a::Vector{Float64}
    b::Vector{Float64}
    c0::Vector{ComplexF64}
    sgn::Float64

    function AHOProgram(ω, λ, T, K, sgn)
        osc = CONCAVE.Hamiltonians.Oscillator(ω, λ)
        ham = CONCAVE.Hamiltonians.Hamiltonian(osc)
        ψ₀ = zero(ham.F.vectors[:,1])
        ψ₀[1:5] .= [0., -1.0im, 1., 0.25im, 2.0]
        ψ₀ = ψ₀ / (ψ₀'ψ₀)

        # Construct algebra, Hamiltonian, et cetera
        I,x,p,an = let
            I,c = BosonAlgebra()
            x = sqrt(1/(2*ω)) * (c + c')
            p = 1im * sqrt(ω/2) * (c' - c)
            I,x,p,c
        end
        H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4
        gens = [I, x, p, x^2, p^2, x*p]
        N = length(gens)
        basis = []
        for g in gens, g′ in gens
            for b in keys((g*g′).terms)
                if !(b in basis)
                    push!(basis, b)
                end
            end
        end
        # The matrix of operators
        M = Matrix{BosonOperator}(undef, length(gens), length(gens))
        for (i,g) in enumerate(gens)
            for (j,g′) in enumerate(gens)
                M[i,j] = g' * g
            end
        end
        # Degrees of freedom, and matrices for plucking out expectation values
        m,E = let
            m = Dict{Boson, Matrix{ComplexF64}}()
            E = Dict{Boson, Matrix{ComplexF64}}()
            for op in basis
                mat = zeros(ComplexF64, (N,N))
                for i in 1:length(gens), j in 1:length(gens)
                    mat[i,j] += M[i,j][op]
                end
                E[op] = mat' / tr(mat'mat)
                if op != Boson(0,0)
                    m[op] = mat
                end
            end
            m,E
        end

        # Algebraic identities
        A,a = let
            A = Matrix{ComplexF64}[]
            b = Float64[]
            for i in 1:(length(gens)^2-length(m))
                # Generate random Hermitian matrix.
                mat = randn(ComplexF64, (length(gens),length(gens)))
                mat = mat + mat'
                # Orthogonalize against A and m
                for a in Iterators.flatten([A,values(m)])
                    mat -= a * (tr(mat * a')) / (tr(a * a'))
                end
                # Normalize
                mat = mat / sqrt(tr(mat' * mat))
                push!(A, mat)
                push!(b, 0.)
            end

            # Identity constraint
            mat = zeros(ComplexF64, (length(gens), length(gens)))
            mat[1,1] = 1.
            push!(A, mat)
            push!(b, 1.)
            A,b
        end

        # Boundary conditions
        B,b = let
            B = Matrix{ComplexF64}[]
            b = Float64[]
            for bos in basis
                ψ = ψ₀
                for n in 1:bos.an
                    ψ = ham.op["a"]*ψ
                end
                for n in 1:bos.cr
                    ψ = ham.op["a"]'*ψ
                end
                val = ψ₀'ψ
                mat = E[bos]
                matp = conj(transpose(mat))
                # Hermitian part
                push!(B, (mat+matp)/2)
                push!(b, real(ψ₀'ψ))
                # Anti-Hermitian part
                push!(B, -1im * (mat-matp)/2)
                push!(b, imag(ψ₀'ψ))
            end
            B,b
        end

        # Equations of motion
        C,D,c0 = let
            # C is the matrix that plucks out the thing to be differentiated. D is
            # the matrix that selects the derivative.
            C = Matrix{ComplexF64}[]
            D = Matrix{ComplexF64}[]
            c0 = ComplexF64[]
            for b in basis
                op = Operator(b)
                dop = 1im * (H*op - op*H)
                mat = zeros(ComplexF64, length(gens), length(gens))
                ok = true
                for top in keys(dop.terms)
                    if top in keys(E)
                        mat += dop[top] * E[top]
                    else
                        ok = false
                    end
                end
                if ok
                    push!(C, E[b])
                    push!(D, mat)
                    ψ = ψ₀
                    O = zero(ham.H)
                    for i in 1:size(O)[1]
                        O[i,i] = 1.0 + 0.0im
                    end
                    for i in 1:b.cr
                        O = O * an'
                    end
                    for i in 1:b.an
                        O = O * an
                    end
                    psi = O*ψ
                    push!(c0, ψ₀'ψ)
                end
            end
            C,D,c0
        end

        O = zeros(ComplexF64, (N,N))
        for b in keys(x.terms)
            O .+= x[b] * E[b]
        end

        new(T,K,N,O,A,B,C,D,a,b,c0,sgn)
    end
end

function size(p::AHOProgram)::Int
    # At the initial time, there is one free parameter for each boundary value.
    # At the late time, there are no free parameters.
    r = length(p.B)
    # In between, a quadratic spline.  The parameter K counts the number of
    # knots.
    r1 = length(p.A) + length(p.C)
    r += (3 + p.K) * r1
    return r
end

function initial(p::AHOProgram)::Vector{Float64}
    return rand(Float64, size(p))
end

function objective!(g, p::AHOProgram, y::Vector{Float64})::Float64
    g .= 0.0
    r = 0.0
    y = y[1+length(p.B):end]
    # Algebra integrals
    for (i,ai) in enumerate(p.a)
        a, b, c, y = y[1], y[2], y[3:3+p.K], y[4+p.K:end]
        r += -ai * iqspline(p.T, a, b, c)
    end
    # Boundary values
    for (k,C) in enumerate(p.C)
        a, b, c, y = y[1], y[2], y[3:3+p.K], y[4+p.K:end]
        λ = qspline(p.T, p.T, a, b, c)
        r += λ * p.c0[k]
    end
    return r
end

# Evaluate quadratic spline.
function qspline(t::Float64, T::Float64, a::Float64, b::Float64, c::Vector{Float64})::Float64
    K = length(c)-1
    f = a
    f′ = b
    f′′ = c[1]
    c = c[2:end]
    dt = T/(K+1)
    for k in 1:K
        # The time of the knot.
        tk = k * T/(K+1)
        if t < tk
            # Time after last knot
            t′ = t - tk + dt
            return f + t′ * f′ + 0.5 * t′^2 * f′′
        end
        f = f + dt*f′
        f′ = f′ + 0.5 * dt^2 * f′′
    end
    t′ = t - T + dt
    return f + t′ * f′ + 0.5 * t′^2 * f′′
end

# Evaluate the derivative of a quadratic spline.
function dqspline(t::Float64, T::Float64, a::Float64, b::Float64, c::Vector{Float64})::Float64
    K = length(c)-1
    f = a
    f′ = b
    f′′ = c[1]
    c = c[2:end]
    dt = T/(K+1)
    for k in 1:K
        # The time of the knot.
        tk = k * T/(K+1)
        if t < tk
            # Time after last knot
            t′ = t - tk + dt
            return f′ + t′ * f′′
        end
        f = f + dt*f′
        f′ = f′ + 0.5 * dt^2 * f′′
    end
    t′ = t - T + dt
    return t′ * f′ + t′ * f′′
end

# Integrate a quadratic spline
function iqspline(T::Float64, a::Float64, b::Float64, c::Vector{Float64})::Float64
    K = length(c)-1
    f = a
    f′ = b
    f′′ = c[1]
    c = c[2:end]
    dt = T/(K+1)
    # Integral to the first knot.
    r = dt * f + dt^2 * f′ / 2 + dt^3 * f′′ / 6
    for k in 1:K
        # Add the integral to the next knot.
        f = f + dt*f′
        f′ = f′ + 0.5 * dt^2 * f′′
        r += dt * f + dt^2 * f′ / 2 + dt^3 * f′′ / 6
    end
    return r
end

function Λ!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64}, t::Float64)::Matrix{ComplexF64}
    # dΛ has shape (N,N,size(p))
    dΛ .= 0.
    # Compute all coefficients with qsplit.
    Λ = zeros(ComplexF64, (p.N,p.N))
    # The "initial" value---at the late time T
    Λ .+= p.O
    y = y[1+length(p.B):end]
    for (i,A) in enumerate(p.A)
        a, b, c, y = y[1], y[2], y[3:3+p.K], y[4+p.K:end]
        λ, da, db, dc = qspline((p.T-t), p.T, a, b, c)
        Λ .+= λ * A
        # TODO dΛ
        #dΛ[:,:,i]
    end
    for (i,C) in enumerate(p.C)
        D = p.D[i]
        a, b, c, y = y[1], y[2], y[3:3+p.K], y[4+p.K:end]
        λ, da, db, dc = qspline((p.T-t), p.T, a, b, c)
        λ′, da′, db′, dc′ = dqspline((p.T-t), p.T, a, b, c)
        Λ .+= λ * D
        Λ .+= λ′ * C
        # TODO dΛ
    end
    return Λ
end

#=
function Λ0!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64})::Matrix{ComplexF64}
    dΛ .= 0.
    Λ = zeros(ComplexF64, (p.N,p.N))
    # TODO
    return Λ
end

function ΛT!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64})::Matrix{ComplexF64}
    dΛ .= 0.
    Λ = zeros(ComplexF64, (p.N,p.N))
    # TODO
    return Λ
end
=#

function constraints!(cb, p::AHOProgram, y::Vector{Float64})
    dΛ = zeros(ComplexF64, (p.N, p.N, size(p)))
    # Spline positivity
    for t in 0:0.01:p.T
        Λ = Λ!(dΛ, p, y, t)
        cb(Λ, dΛ)
    end
    if false
        # Endpoint positivity
        Λ = Λ0!(dΛ, p, y)
        cb(Λ, dΛ)
        Λ = ΛT!(dΛ, p, y)
        cb(Λ, dΛ)
    end
end

function demo(::Val{:RT})
    # Parameters.
    ω = 1.
    λ = 1.0
    T = 5.0
    K = 2

    # For diagonalizing.
    dt = 1e-1
    p = CONCAVE.Hamiltonians.Oscillator(ω, λ)
    ham = CONCAVE.Hamiltonians.Hamiltonian(p)
    Ω = ham.F.vectors[:,1]
    ψ = zero(Ω)
    #ψ[1:5] .= [0., -1.0im, 1., 1.0im, 0.0]
    ψ[1:5] .= [0., -1.0im, 1., 0.25im, 2.0]
    ψ₀ = copy(ψ)
    U = CONCAVE.Hamiltonians.evolution(ham, dt)

    for t in 0:dt:T
        ex = real(ψ' * ham.op["x"] * ψ)

        plo = AHOProgram(ω, λ, T, K, 1.0)
        phi = AHOProgram(ω, λ, T, K, -1.0)
        lo, ylo = CONCAVE.IPM.solve(plo; verbose=false)
        hi, yhi = CONCAVE.IPM.solve(phi; verbose=false)

        println("$t $ex $(-lo) $hi")
        ψ = U * ψ
    end
end

function demo(::Val{:SpinRT})
    # Parameters.
    J = 1.

    # Construct operators.
    I,X,Y,Z = SpinAlgebra()

    # Build the Hamiltonian.
end

function demo(::Val{:ScalarRT})
    # Parameters
    N = 5

    # Construct operators.

    # Build the Hamiltonian.
end

struct HubbardRTProgram
end

function size(p::HubbardRTProgram)::Int
    return 3
end

function initial(p::HubbardRTProgram)::Vector{Float64}
    return rand(Float64, size(p))
end

function objective!(g, p::HubbardRTProgram, y::Vector{Float64})::Float64
end

function constraints!(cb, p::HubbardRTProgram, y::Vector{Float64})
end

function demo(::Val{:HubbardRT})
    # Parameters
    N = 10

    # Construct operators

    # Build the Hamiltonian.
end

function demo(::Val{:Neutrons})
    # Parameters.

    # Construct operators.

    # Build the Hamiltonian.
end

function demo(::Val{:Thermo})
end

function demo(::Val{:SpinThermo})
end

function demo(::Val{:Coulomb})
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "--demo"
            arg_type = Symbol
        end
        parse_args(s)
    end

    if !isnothing(args["demo"])
        demo(args["demo"])
        return
    end
end

main()

