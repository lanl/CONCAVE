using ArgParse
using LinearAlgebra: tr

using CONCAVE

import CONCAVE.Programs: initial, constraints!, objective!

demo(s::Symbol) = demo(Val(s))

struct AHOProgram <: ConvexProgram
    T::Float64
    K::Int
    N::Int
    A::Vector{Matrix{ComplexF64}}
    B::Vector{Matrix{ComplexF64}}
    C::Vector{Matrix{ComplexF64}}
    D::Vector{Matrix{ComplexF64}}
    a::Vector{Float64}
    b::Vector{Float64}
    sgn::Float64

    function AHOProgram(ω, λ, T, K, sgn)
        osc = CONCAVE.Hamiltonians.Oscillator(ω, λ)
        ham = CONCAVE.Hamiltonians.Hamiltonian(osc)
        ψ₀ = zero(ham.F.vectors[:,1])
        ψ₀[1:5] .= [0., -1.0im, 1., 0.25im, 2.0]
        ψ₀ = ψ₀ / (ψ₀'ψ₀)

        # Construct algebra, Hamiltonian, et cetera
        I,x,p = let
            I,c = BosonAlgebra()
            x = sqrt(1/(2*ω)) * (c + c')
            p = 1im * sqrt(ω/2) * (c' - c)
            I,x,p
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
            # Initial conditions, determined quasi-manually.
            initial = Dict{Boson, ComplexF64}()
            for i in 0:12, j in 0:12
                ψ = ψ₀
                for n in 1:j
                    ψ = ham.op["a"]*ψ
                end
                for n in 1:i
                    ψ = ham.op["a"]'*ψ
                end
                initial[Boson(i,j)] = ψ₀'ψ
            end

            B = Matrix{ComplexF64}[]
            b = Float64[]
            # TODO
            B,b
        end

        # Equations of motion
        C,D = let
            # C is the matrix that plucks out the thing to be differentiated. D is
            # the matrix that selects the derivative.
            C = Matrix{ComplexF64}[]
            D = Matrix{ComplexF64}[]
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
                end
            end
            C,D
        end

        new(T,K,N,A,B,C,D,a,b,sgn)
    end
end

function size(p::AHOProgram)::Int
    # TODO
    return 3
end

function initial(p::AHOProgram)::Vector{Float64}
    return rand(Float64, size(p))
end

function objective!(g, p::AHOProgram, y::Vector{Float64})::Float64
    g .= 0.0
    r = 0.0
    # TODO
    return r
end

function Λ!(dΛ::Array{ComplexF64,3}, p::AHOProgram, y::Vector{Float64})::Matrix{ComplexF64}
    Λ = zeros(ComplexF64, (p.N,p.N))
    # TODO
    return Λ
end

function constraints!(cb, p::AHOProgram, y::Vector{Float64})
    dΛ = zeros(ComplexF64, (p.N, p.N, size(p)))
    for t in 0:0.01:p.T
        Λ = Λ!(dΛ, p, y)
        # TODO probably we should just be passing Λ into cb, right? cb should
        # be a function, not a method...
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

