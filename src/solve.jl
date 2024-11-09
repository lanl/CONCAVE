using ArgParse
using LinearAlgebra
using Profile
using REPL

using CONCAVE
using CONCAVE.Splines
using CONCAVE.Utilities: check_gradients, print_mathematica

import Base: size
import CONCAVE.Programs: initial, constraints!, objective!

demo(s::Symbol; verbose=false) = demo(Val(s), verbose)

const SLACK::Float64 = 0e-4

function aho_state_initialize!(ψ)
    ψ .= 0.0
    #ψ[1:5] .= [0., -1.0im, 1., 0.25im, 2.0]
    ψ[1:3] .= [1.0, 0.5, 0.25]
    ψ .= ψ / sqrt(ψ'ψ)
end

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

    function AHOProgram(ω, λ, T, K, N, sgn)
        osc = CONCAVE.Hamiltonians.Oscillator(ω, λ)
        ham = CONCAVE.Hamiltonians.Hamiltonian(osc)
        ψ₀ = zero(ham.F.vectors[:,1])
        aho_state_initialize!(ψ₀)

        # Construct algebra, Hamiltonian, et cetera
        I,x,p,an = let
            I,c = BosonAlgebra()
            x = sqrt(1/(2*ω)) * (c + c')
            p = 1im * sqrt(ω/2) * (c' - c)
            I,x,p,c
        end
        H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4
        gens = [I, x, p, x^2, x*p, x^3, x*x*p, p^2, x^4]
        gens = gens[1:N]
        basis = []
        for g in gens, g′ in gens
            pr = g′' * g
            dpr = 1im * (H * pr - pr * H)
            for b in keys(pr.terms) ∪ keys(dpr.terms)
                #if abs(pr[b]) < 1e-10 && abs(dpr[b]) < 1e-10
                #    continue
                #end
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
                    iprod::ComplexF64 = 0.
                    nrm::Float64 = 0.
                    for b in keys(o.terms) ∪ keys(o′.terms)
                        iprod += conj(o[b]) * o′[b]
                    end
                    for b in keys(o′.terms)
                        nrm += abs(o′.terms[b])^2
                    end
                    o = o - iprod*o′ / nrm
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
                        for k in 1:b.an
                            ψ = ham.op["a"] * ψ
                        end
                        for k in 1:b.cr
                            ψ = ham.op["a"]' * ψ
                        end
                        M0[i,j] += c*ψ₀'ψ
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
                    if sum(abs.(mat)) ≥ 1e-10
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

        # Equations of motion.
        C,D,c0,λT = let
            C = Matrix{ComplexF64}[]
            D = Matrix{ComplexF64}[]
            c0 = Float64[]
            Cop = []
            xops = []
            yops = []
            Es = []

            # Construct a list of operators and extractors.
            for i in 1:N
                for j in 1:i
                    op₊ = 0.5 * (M[i,j] + M[j,i])
                    op₋ = 0.5im * (M[i,j] - M[j,i])
                    if independent(op₊, xops)
                        E = zeros(ComplexF64, (N,N))
                        E[i,j] += 0.5
                        E[j,i] += 0.5
                        push!(xops, op₊)
                        push!(Es, E)
                    end
                    if independent(op₋, xops)
                        E = zeros(ComplexF64, (N,N))
                        E[i,j] -= 0.5im
                        E[j,i] += 0.5im
                        push!(xops, op₋)
                        push!(Es, E)
                    end
                end
            end

            # Construct a list of "untracked" operators.
            for op in xops
                dop = 1im * (H * op - op * H)
                if independent(dop, xops ∪ yops)
                    push!(yops, dop)
                end
            end

            Nx = length(xops)
            Ny = length(yops)

            # Construct derivative matrices
            d = zeros(Float64, (Nx,Nx))
            d̃ = zeros(Float64, (Nx,Ny))
            for (i,op) in enumerate(xops)
                dop = 1im * (H * op - op * H)
                v = zeros(ComplexF64, length(basis))
                F = zeros(ComplexF64, (length(basis),Nx+Ny))
                @assert keys(dop.terms) ⊆ basis
                for (k,b) in enumerate(basis)
                    v[k] = dop[b]
                    for (k′,op′) in enumerate(xops)
                        F[k,k′] = op′[b]
                    end
                    for (k′,op′) in enumerate(yops)
                        F[k,Nx+k′] = op′[b]
                    end
                end
                u = F \ v
                @assert maximum(imag.(u)) < 1e-8

                d[i,:] = real(u[1:Nx])
                d̃[i,:] = real(u[Nx+1:Nx+Ny])
            end

            # Orthonormalize the columns of d̃.
            d̃s = []
            for i in 1:Ny
                v = d̃[:,i]
                for u in d̃s
                    v = v - (v⋅u)*u
                end
                v /= sqrt(v⋅v)
                push!(d̃s,v)
            end

            # Create an orthogonal set of degrees of freedom.
            vs = []
            for i in 1:Nx
                v = randn(Float64, Nx)
                # Orthogonalize against previous vectors.
                for u in vs
                    v = v - (v⋅u)*u
                end

                # Orthogonalize against columns of d̃.
                for j in 1:Ny
                    u = d̃s[j]
                    v = v - (v⋅u)*u
                end

                # Normalize
                if abs(v⋅v) ≤ 1e-8
                    break
                end
                v /= sqrt(v⋅v)

                push!(vs, v)
            end

            for (i,v) in enumerate(vs)
                op = zero(Operator{Boson})
                for (k,xop) in enumerate(xops)
                    op += v[k] * xop
                end
                # Find Cmat
                Cmat = let
                    w = zeros(ComplexF64, length(basis))
                    F = zeros(ComplexF64, (length(basis),Nx))
                    for (k,b) in enumerate(basis)
                        w[k] = op[b]
                        for (k′,op′) in enumerate(xops)
                            F[k,k′] = op′[b]
                        end
                    end
                    u = F \ w
                    mat = zeros(ComplexF64, (N,N))
                    for j in 1:Nx
                        mat += u[j] * Es[j]
                    end
                    mat
                end

                # Find Dmat
                dop′ = 1im * (H*op - op*H)
                dop = zero(Operator{Boson})
                for (k,xop) in enumerate(xops)
                    dop += (v' * d)[k] * xop
                end
                Dmat = let
                    w = zeros(ComplexF64, length(basis))
                    F = zeros(ComplexF64, (length(basis),Nx))
                    for (k,b) in enumerate(basis)
                        w[k] = dop[b]
                        for (k′,op′) in enumerate(xops)
                            F[k,k′] = op′[b]
                        end
                    end
                    u = F \ w
                    mat = zeros(ComplexF64, (N,N))
                    for j in 1:Nx
                        mat += u[j] * Es[j]
                    end
                    mat
                end

                # Add derivative relation
                push!(Cop, op)
                push!(C, Cmat)
                push!(D, Dmat)
                # Add initial value
                push!(c0, real(tr(Cmat * M0)))
            end

            # Spline coefficients
            O = sgn * x
            λT = let
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
            for (k,a) in enumerate(A)
                print("a[$k] = ")
                print_mathematica(a)
                println()
            end
            for (k,c) in enumerate(C)
                print("c[$k] = ")
                print_mathematica(c)
                println()
            end
            for (k,d) in enumerate(D)
                print("d[$k] = ")
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
    if !isnothing(g)
        g .= 0.0
    end
    r::Float64 = 0.0
    spline = QuadraticSpline(p.T, p.K)
    o::Int = 0
    # Run up the offset
    for (i,A) in enumerate(p.A)
        o += 3+p.K
    end
    # Boundary values
    for (k,C) in enumerate(p.C)
        spline.c[1] = p.λT[k]
        spline.c[2:end] = y[1+o:2+p.K+o]
        at!(spline, p.T)
        r += spline.f * p.c0[k]
        if !isnothing(g)
            for (j,∂) in enumerate(spline.∂c[2:end])
                g[o+j] += p.c0[k] * ∂
            end
        end
        o += 2+p.K
    end

    r *= -1
    if !isnothing(g)
        g .*= -1
    end
    return r
end

function objective!(g, h, p::AHOProgram, y::Vector{Float64})::Float64
    if !isnothing(h)
        h .= 0.0
    end
    return objective!(g, p, y)
end

function Λ!(dΛ, p::AHOProgram, y::Vector{Float64}, t::Float64)::Matrix{ComplexF64}
    if !isnothing(dΛ)
        # dΛ has shape (N,N,size(p))
        dΛ .= 0.
    end
    spline = QuadraticSpline(p.T, p.K)
    Λ::Matrix{ComplexF64} = zeros(ComplexF64, (p.N,p.N))
    o::Int = 0
    @views for (i,A) in enumerate(p.A)
        spline.c[1:end] .= y[1+o:3+p.K+o]
        at!(spline, p.T-t)
        Λ .+= spline.f .* A
        if !isnothing(dΛ)
            for (j,∂) in enumerate(spline.∂c[1:end])
                dΛ[:,:,j+o] .+= A .* ∂
            end
        end
        o += 3+p.K
    end
    @views for (i,C) in enumerate(p.C)
        D = p.D[i]
        spline.c[1] = p.λT[i]
        spline.c[2:end] .= y[1+o:2+p.K+o]
        at!(spline, p.T-t)
        Λ .+= spline.f .* D
        Λ .-= spline.f′ .* C # My spline has t reversed
        if !isnothing(dΛ)
            for j in 2:length(spline.∂c)
                ∂ = spline.∂c[j]
                ∂′ = spline.∂c′[j]
                dΛ[:,:,j+o-1] .+= ∂ .* D
                dΛ[:,:,j+o-1] .-= ∂′ .* C
            end
        end
        o += 2+p.K
    end
    return Λ
end

function constraints!(cb, p::AHOProgram, y::Vector{Float64})
    dΛ = zeros(ComplexF64, (p.N, p.N, size(p)))
    # Spline positivity
    for t in LinRange(0,p.T,1 + 10*(1+p.K))
        Λ = Λ!(dΛ, p, y, t)
        cb(Λ + SLACK * I, dΛ, 0)
    end
end

function demo(::Val{:RT}, verbose)
    # Parameters.
    ω = 1.
    λ = 1.0
    T = 5.0
    T = 2.0 # TODO

    # For diagonalizing.
    dt = 1e-1
    p = CONCAVE.Hamiltonians.Oscillator(ω, λ)
    ham = CONCAVE.Hamiltonians.Hamiltonian(p)
    Ω = ham.F.vectors[:,1]

    if false
        # Check gradients!
        N,K = 4,2
        plo = AHOProgram(ω, λ, T, K, N, 1.0)
        y = CONCAVE.IPM.feasible_initial(plo)
        g = zero(y)
        h = zeros(Float64, (length(y),length(y)))
        # Objective gradients.
        CONCAVE.IPM.objective!(g, plo, y)
        @assert check_gradients(y, g, nothing; verbose=true) do y
            CONCAVE.IPM.objective!(nothing, plo, y)
        end
        # Barrier gradients.
        CONCAVE.IPM.barrier!(g, h, plo, y)
        @assert check_gradients(y, g, nothing; verbose=true) do y
            CONCAVE.IPM.barrier!(nothing, nothing, plo, y)
        end
        exit(0)
    end

    ψ = zero(Ω)
    aho_state_initialize!(ψ)
    ψ₀ = copy(ψ)
    U = CONCAVE.Hamiltonians.evolution(ham, dt)
    for t in 0.0:dt:T
        ex = real(ψ' * ham.op["x"] * ψ)
        println("$t -1 -1 $ex $ex")
        ψ = U*ψ
    end

    #for (N,K) in [(4,0), (4,5)]
    for (N,K) in [(4,0), (4,3), (9,0), (9,3)]
    #for (N,K) in [(9,3)]
        p0 = AHOProgram(ω, λ, 0.0, K, N, 1.0)
        printstyled(stderr, "N = $N; K = $K\n", bold=true)
        printstyled(stderr, "Algebraic constraints: $(length(p0.A))\n", bold=true)
        printstyled(stderr, "Derivatives: $(length(p0.C))\n", bold=true)
        printstyled(stderr, "Parameters: $(size(p0))\n", bold=true)
        for t in dt:dt:T
            plo = AHOProgram(ω, λ, t, K, N, 1.0)
            phi = AHOProgram(ω, λ, t, K, N, -1.0)

            lo, ylo = CONCAVE.IPM.solve(plo; verbose=verbose)
            hi, yhi = CONCAVE.IPM.solve(phi; verbose=verbose)

            if -lo > hi
                println(stderr, "WARNING: primal proved infeasible")
            end

            println("$t $N $K $(-lo) $hi")
            flush(stdout)
         end
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

function demo(::Val{:Hubbard}, verbose)
end

function demo(::Val{:Coulomb}, verbose)
end

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
            "--profile"
                action = :store_true
            "--demo"
                arg_type = Symbol
            "-v","--verbose"
                action = :store_true
        end
        parse_args(s)
    end

    if args["profile"]
        ω = 1.0
        λ = 1.0
        T = 1.0
        K = 1
        p = AHOProgram(ω, λ, T, K, 6, 1.0)
        @profile CONCAVE.IPM.solve(p)
        open("prof-flat", "w") do f
            Profile.print(f, format=:flat, sortedby=:count)
        end
        open("prof-tree", "w") do f
            Profile.print(f, noisefloor=2.0)
        end
    end

    if !isnothing(args["demo"])
        demo(args["demo"]; verbose=args["verbose"])
        return
    end
end

main()

