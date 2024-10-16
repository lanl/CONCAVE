using ArgParse
using LinearAlgebra: tr

using CONCAVE
using CONCAVE.Splines

import Base: size
import CONCAVE.Programs: initial, constraints!, objective!

demo(s::Symbol; verbose=false) = demo(Val(s), verbose)

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
        gens = [I, x, p, x^2]
        #gens = [I, x, p, x^2, p^2, x*p]
        N = length(gens)
        basis = []
        for g in gens, g′ in gens
            p = g′' * g
            for b in keys(p.terms)
                if p[b] ≈ 0
                    continue
                end
                if !(b in basis)
                    push!(basis, b)
                end
            end
        end

        # The matrix of operators
        M = Matrix{BosonOperator}(undef, length(gens), length(gens))
        for (i,g) in enumerate(gens)
            for (j,g′) in enumerate(gens)
                M[i,j] = g' * g′
            end
        end
        # Degrees of freedom, and matrices for plucking out expectation values
        m′,E = let
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
        # Hermitian basis for the degrees of freedom.
        m = let
            m = Matrix{ComplexF64}[]
            for mat′ in values(m′)
                # Hermitize
                for mat in [0.5 * (mat′' + mat′), 0.5im * (mat′' - mat′)]
                    # Orthogonalize
                    for a in m
                        mat -= a * tr(mat * a') / tr(a * a')
                    end
                    if (sum(abs.(mat))) ≉ 0
                        push!(m, mat)
                    end
                end
            end
            m
        end

        # Algebraic identities
        A,a = let
            A = Matrix{ComplexF64}[]
            b = Float64[]
            # Identity constraint
            mat = zeros(ComplexF64, (length(gens), length(gens)))
            mat[1,1] = 1.
            push!(A, mat)
            push!(b, 1.)

            for i in 1:(length(gens)^2-length(m)-1)
                # Generate random Hermitian matrix.
                mat = randn(ComplexF64, (length(gens),length(gens)))
                mat = mat + mat'
                # Orthogonalize against A and m
                for a in Iterators.flatten([A,values(m)])
                    mat -= a * tr(mat * a') / tr(a * a')
                end
                # Normalize
                mat = mat / sqrt(tr(mat' * mat))
                push!(A, mat)
                push!(b, 0.)
            end

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
    return length(p.A) * (3 + p.K) + length(p.C) * (2 + p.K)
end

function initial(p::AHOProgram)::Vector{Float64}
    return rand(Float64, size(p))
end

function objective!(g, p::AHOProgram, y::Vector{Float64})::Float64
    g .= 0.0
    r = 0.0
    spline = QuadraticSpline(p.T, p.K)
    o::Int = 0
    # Algebra integrals
    for (i,ai) in enumerate(p.a)
        spline.c[1:end] = y[1+o:3+p.K+o]
        at!(spline, p.T)
        r += -ai * spline.∫
        for (j,∂) in enumerate(spline.∂∫[1:end])
            g[o+j] += -ai*∂
        end
        o += 3+p.K
    end
    # Boundary values
    for (k,C) in enumerate(p.C)
        spline.c[2:end] = y[1+o:2+p.K+o]
        at!(spline, p.T)
        r += spline.f * p.c0[k]
        for (j,∂) in enumerate(spline.∂c[2:end])
            g[o+j] += p.c0[k] * ∂
        end
        o += 2+p.K
    end
    return r
end

function Λ!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64}, t::Float64)::Matrix{ComplexF64}
    # dΛ has shape (N,N,size(p))
    dΛ .= 0.
    spline = QuadraticSpline(p.T, p.K)
    Λ = zeros(ComplexF64, (p.N,p.N))
    # The "initial" value---at the late time T
    Λ .+= p.O
    o::Int = 0
    for (i,A) in enumerate(p.A)
        spline.c[1:end] = y[1+o:3+p.K+o]
        at!(spline, p.T-t)
        Λ .+= spline.f * A
        for (j,∂) in enumerate(spline.∂c[1:end])
            dΛ[:,:,j+o] .+= A * ∂
        end
        o += 3+p.K
    end
    for (i,C) in enumerate(p.C)
        D = p.D[i]
        spline.c[2:end] = y[1+o:2+p.K+o]
        at!(spline, p.T-t)
        Λ .+= spline.f * D
        Λ .+= spline.f′ * C
        for j in 2:length(spline.∂c)
            ∂ = spline.∂c[j]
            ∂′ = spline.∂c′[j]
            dΛ[:,:,j+o-1] .+= ∂ * D
            dΛ[:,:,j+o-1] .+= ∂′ * C
        end
        o += 2+p.K
    end
    return Λ
end

function constraints!(cb, p::AHOProgram, y::Vector{Float64})
    dΛ = zeros(ComplexF64, (p.N, p.N, size(p)))
    # Spline positivity
    for t in 0:0.01:p.T
        Λ = Λ!(dΛ, p, y, t)
        cb(Λ, dΛ)
    end
end

function demo(::Val{:RT}, verbose)
    # Parameters.
    ω = 1.
    λ = 1.0
    T = 5.0
    K = 0

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

    if false
        plo = AHOProgram(ω, λ, T, K, 1.0)
        phi = AHOProgram(ω, λ, T, K, -1.0)
        p1 = CONCAVE.IPM.Phase1(phi)
        z = initial(p1)
        @assert CONCAVE.IPM.feasible(p1, z)
        ϵ = 1e-4
        g = zero(z)
        g′ = zero(z)
        bar = CONCAVE.IPM.barrier!(g, p1, z)
        for i in 1:length(z)
            z₊ = zero(z)
            z₊ .= z
            z₊[i] += ϵ
            z₋ = zero(z)
            z₋ .= z
            z₋[i] -= ϵ
            bar₊ = CONCAVE.IPM.barrier!(g′, p1, z₊)
            bar₋ = CONCAVE.IPM.barrier!(g′, p1, z₋)
            gest = (bar₊ - bar₋)/(2*ϵ)
            rerr = abs((gest-g[i])/(gest+1e-4))
            println(gest, "     ", g[i], "    ", rerr)
            if rerr > 1e-4
                println("WARNING")
            end
        end
        exit(0)
    end

    if false
        # Check phase-2 derivatives (barrier).
        plo = AHOProgram(ω, λ, T, K, 1.0)
        y = CONCAVE.IPM.feasible_initial(plo)
        ϵ = 1e-4
        g = zero(y)
        g′ = zero(y)
        bar = CONCAVE.IPM.barrier!(g, plo, y)
        for i in 1:length(y)
            y₊ = zero(y)
            y₊ .= y
            y₊[i] += ϵ
            y₋ = zero(y)
            y₋ .= y
            y₋[i] -= ϵ
            bar₊ = CONCAVE.IPM.barrier!(g′, plo, y₊)
            bar₋ = CONCAVE.IPM.barrier!(g′, plo, y₋)
            gest = (bar₊ - bar₋)/(2*ϵ)
            rerr = abs((gest-g[i])/(gest + 1e-4))
            println(gest, "     ", g[i], "    ", rerr)
            if rerr > 1e-4
                println("WARNING")
            end
        end
        exit(0)
    end

    if false
        # Check phase-2 derivatives (objective).
        plo = AHOProgram(ω, λ, T, K, 1.0)
        y = CONCAVE.IPM.feasible_initial(plo)
        ϵ = 1e-4
        g = zero(y)
        g′ = zero(y)
        obj = CONCAVE.IPM.objective!(g, plo, y)
        for i in 1:length(y)
            y₊ = zero(y)
            y₊ .= y
            y₊[i] += ϵ
            y₋ = zero(y)
            y₋ .= y
            y₋[i] -= ϵ
            obj₊ = CONCAVE.IPM.objective!(g′, plo, y₊)
            obj₋ = CONCAVE.IPM.objective!(g′, plo, y₋)
            gest = (obj₊ - obj₋)/(2*ϵ)
            println(gest, "     ", g[i], "    ", (gest-g[i])/(1e-5 + gest))
            if (gest-g[i]) / gest > 1e-4
                println("WARNING")
            end
        end
        exit(0)
    end

    for t in dt:dt:T
        ψ = U*ψ
        ex = real(ψ' * ham.op["x"] * ψ)

        plo = AHOProgram(ω, λ, T, K, 1.0)
        phi = AHOProgram(ω, λ, T, K, -1.0)

        if verbose
            println("Algebraic constraints: ", length(plo.A))
            println("Derivatives: ", length(plo.C))
        end

        lo, ylo = CONCAVE.IPM.solve(plo; verbose=true)
        hi, yhi = CONCAVE.IPM.solve(phi; verbose=true)

        if -lo > hi
            println(stderr, "WARNING: primal proved infeasible")
        end

        println("$t $ex $(-lo) $hi")
        ψ = U * ψ
    end
end

function demo(::Val{:SpinRT}, verbose)
    # Parameters.
    J = 1.

    # Construct operators.
    I,X,Y,Z = SpinAlgebra()

    # Build the Hamiltonian.
end

function demo(::Val{:ScalarRT}, verbose)
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

function demo(::Val{:HubbardRT}, verbose)
    # Parameters
    N = 10

    # Construct operators

    # Build the Hamiltonian.
end

function demo(::Val{:Neutrons}, verbose)
    # Parameters.

    # Construct operators.

    # Build the Hamiltonian.
end

function demo(::Val{:Thermo}, verbose)
end

function demo(::Val{:SpinThermo}, verbose)
end

function demo(::Val{:Coulomb}, verbose)
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "--demo"
                arg_type = Symbol
            "-v","--verbose"
                action = :store_true
        end
        parse_args(s)
    end

    if !isnothing(args["demo"])
        demo(args["demo"]; verbose=args["verbose"])
        return
    end
end

main()

