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
    A::Vector{Matrix{ComplexF64}}
    C::Vector{Matrix{ComplexF64}}
    D::Vector{Matrix{ComplexF64}}
    c0::Vector{Float64}
    λT::Vector{Float64}
    sgn::Float64

    function AHOProgram(ω, λ, T, K, sgn)
        osc = CONCAVE.Hamiltonians.Oscillator(ω, λ)
        ham = CONCAVE.Hamiltonians.Hamiltonian(osc)
        ψ₀ = zero(ham.F.vectors[:,1])
        ψ₀[1:5] .= [0., -1.0im, 1., 0.25im, 2.0]
        ψ₀ = ψ₀ / sqrt(ψ₀'ψ₀)

        # Construct algebra, Hamiltonian, et cetera
        I,x,p,an = let
            I,c = BosonAlgebra()
            x = sqrt(1/(2*ω)) * (c + c')
            p = 1im * sqrt(ω/2) * (c' - c)
            I,x,p,c
        end
        H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4
        gens = [I, x, p, x^2]
        if false
            # TODO
            H = p^2 / 2 + ω^2 * x^2 / 2
            gens = [I, x, p]
        end
        #gens = [I, x, p, x^2, p^2, x*p]
        N = length(gens)
        basis = []
        for g in gens, g′ in gens
            pr = g′' * g
            for b in keys(pr.terms)
                if abs(pr[b]) < 1e-10
                    continue
                end
                if !(b in basis)
                    push!(basis, b)
                end
            end
        end
        # Linearly independent Hermitian basis
        hbasis = []
        for bas in basis
            b = Operator(bas)
            o₊ = b + b'
            o₋ = 1im * (b - b')
            for o in (o₊,o₋)
                for o′ in hbasis
                    ip = 0.
                    nrm = 0.
                    for b in keys(o.terms)
                        if b in keys(o′.terms)
                            ip += conj(o.terms[b]) * o′.terms[b]
                        end
                    end
                    for b in keys(o′.terms)
                        nrm += abs(o′.terms[b])^2
                    end
                    o = o - ip*o′ / nrm
                end
                is0 = true
                for (b,c) in o.terms
                    if abs(c) > 1e-10
                        is0 = false
                    end
                end
                if !is0
                    push!(hbasis, o)
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
        # Expectation values in the initial state
        M0 = let
            M0 = zeros(ComplexF64, (N,N))
            for (i,g) in enumerate(gens)
                for (j,g′) in enumerate(gens)
                    op = g' * g′
                    for (b,c) in op.terms
                        ψ = ψ₀
                        for k in 1:b.cr
                            ψ = ham.op["a"]' * ψ
                        end
                        for k in 1:b.an
                            ψ = ham.op["a"] * ψ
                        end
                        M0[i,j] += c*real(ψ₀'ψ)
                    end
                end
            end
            M0
        end

        # Degrees of freedom.
        m′ = let
            m = Dict{Boson, Matrix{ComplexF64}}()
            for op in basis
                mat = zeros(ComplexF64, (N,N))
                for i in 1:length(gens), j in 1:length(gens)
                    mat[i,j] += M[i,j][op]
                end
                m[op] = mat
            end
            m
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
        A = let
            A = Matrix{ComplexF64}[]
            for i in 1:(length(gens)^2-length(m))
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
            end

            A
        end

        # Equations of motion.
        # TODO this can fail to find all possible EoMs, if there's a linear
        # combination for which untracked coefficients cancel.
        C,D,c0,λT = let
            Cop = []
            C = Matrix{ComplexF64}[]
            D = Matrix{ComplexF64}[]
            c0 = Float64[]

            function ip(o′,o)::ComplexF64
                r::ComplexF64 = 0
                for b in keys(o.terms) ∪ keys(o′.terms)
                    r += conj(o′[b]) * o[b]
                end
                return r
            end

            function independent(o, l)::Bool
                # First orthogonormalize l
                l′ = []
                for o′ in l
                    for o′′ in l′
                        coef = ip(o′, o′′)
                        o′ = o′ - conj(coef) * o′′
                    end
                    if real(ip(o′,o′)) > 1e-8
                        o′ /= sqrt(ip(o′,o′))
                        push!(l′, o′)
                    end
                end
                for o′ in l′
                    coef = ip(o, o′)
                    o = o - conj(coef) * o′
                end
                for b in keys(o.terms)
                    if abs(o[b]) > 1e-8
                        return true
                    end
                end
                return false
            end

            # Construct a list of operators and extractors.
            ops = []
            Es = []
            for i in 1:N
                for j in 1:i
                    op = M[i,j]
                    if independent(op, ops)
                        E = zeros(ComplexF64, (N,N))
                        E[i,j] += 0.5
                        E[j,i] += 0.5
                        push!(ops, op)
                        push!(Es, E)
                    end
                end
            end

            # Derivative relations
            for (i,op) in enumerate(ops)
                dop = 1im * (H*op - op*H)
                if independent(dop, ops)
                    continue
                end
                # Set up and solve a linear system
                v = zeros(ComplexF64, length(basis))
                F = zeros(ComplexF64, (length(basis),length(ops)))
                for (k,b) in enumerate(basis)
                    v[k] = dop[b]
                    for (k′,op′) in enumerate(ops)
                        F[k,k′] = op′[b]
                    end
                end
                u = F \ v
                d = zeros(ComplexF64, (N,N))
                opcheck = 0*op
                for j in 1:length(ops)
                    d += u[j] * Es[j]
                    opcheck += u[j] * ops[j]
                end
                
                # Add derivative relation
                push!(Cop, op)
                push!(C, Es[i])
                push!(D, d)
                # Add initial value
                push!(c0, real(tr(Es[i] * M0)))
            end

            # Spline coefficients
            O = sgn * x  # TODO extra (-1)?
            λT::Vector{Float64} = let
                v = zeros(ComplexF64, length(basis))
                F = zeros(ComplexF64, (length(basis),length(C)))
                for (k,b) in enumerate(basis)
                    v[k] = O[b]
                    for (k′,op) in enumerate(Cop)
                        F[k,k′] = op[b]
                    end
                end
                u = F \ v
                ur = real.(u)
                ui = imag.(u)
                @assert maximum(abs.(ui)) < 1e-8
                ur
            end

            C,D,c0,λT
        end

        if false
            # Output matrices for processing in mathematica
            function print_mathematica(mat)
                print("{")
                for i in 1:N
                    print("{")
                    for j in 1:N
                        print(real(mat[i,j]),"+",imag(mat[i,j]),"I")
                        if j < N
                            print(",")
                        end
                    end
                    print("}")
                    if i < N
                        print(",")
                    end
                end
                print("}")
            end
            for (k,a) in enumerate(A)
                print("A[$k] = ")
                print_mathematica(a)
                println()
            end
            for (k,c) in enumerate(C)
                print("C[$k] = ")
                print_mathematica(c)
                println()
            end
            for (k,d) in enumerate(D)
                print("D[$k] = ")
                print_mathematica(d)
                println()
            end
            exit(0)
        end

        return new(T,K,N,A,C,D,c0,λT,sgn)
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
    r::Float64 = 0.0
    spline = QuadraticSpline(p.T, p.K)
    o::Int = 0
    # Boundary values
    for (k,C) in enumerate(p.C)
        spline.c[1] = p.λT[k]
        spline.c[2:end] = y[1+o:2+p.K+o]
        at!(spline, p.T)
        r += spline.f * p.c0[k]
        for (j,∂) in enumerate(spline.∂c[2:end])
            g[o+j] += p.c0[k] * ∂
        end
        o += 2+p.K
    end

    g .*= -1
    return -r
end

function Λ!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64}, t::Float64)::Matrix{ComplexF64}
    # dΛ has shape (N,N,size(p))
    dΛ .= 0.
    spline = QuadraticSpline(p.T, p.K)
    Λ = zeros(ComplexF64, (p.N,p.N))
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
        spline.c[1] = p.λT[i]
        spline.c[2:end] = y[1+o:2+p.K+o]
        at!(spline, p.T-t)
        Λ .+= spline.f * D
        Λ .-= spline.f′ * C # My spline has t reversed
        for j in 2:length(spline.∂c)
            ∂ = spline.∂c[j]
            ∂′ = spline.∂c′[j]
            dΛ[:,:,j+o-1] .+= ∂ * D
            dΛ[:,:,j+o-1] .-= ∂′ * C
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

    if false
        plo = AHOProgram(ω, λ, T, K, 1.0)
        y = initial(plo)
        y .= 0
        g = zero(y)
        println(length(y))
        y[1] = -1.
        println(CONCAVE.IPM.feasible(plo, y))
        println(CONCAVE.IPM.barrier!(g, plo, y))
        println(CONCAVE.IPM.objective!(g, plo, y))
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

        if false
            dΛ = zeros(ComplexF64, (plo.N, plo.N, size(plo)))
            display(Λ!(dΛ, plo, ylo, plo.T))
            display(Λ!(dΛ, plo, ylo, 0.0))
            exit(0)
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

