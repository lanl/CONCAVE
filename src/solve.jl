using ArgParse
using LinearAlgebra: tr

using CONCAVE

demo(s::Symbol) = demo(Val(s))

function demo(::Val{:RT})
    # Parameters.
    ω = 1.
    λ = 0.1

    # For diagonalizing.
    p = CONCAVE.Hamiltonians.Oscillator(ω, λ)
    ham = CONCAVE.Hamiltonians.Hamiltonian(p)

    # Initial state.
    # TODO

    # Construct operators.
    I,a = BosonAlgebra()
    x = sqrt(1/(2*ω)) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)

    # Build the Hamiltonian.
    H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4

    # Generators of the SDP.
    gens = [I, x, p, x^2, p^2, x*p]

    # Number of time intervals.
    T = 2

    # Basis operators that appear.
    basis = []
    for g in gens
        for g′ in gens
            for b in keys((g*g′).terms)
                if !(b in basis)
                    push!(basis, b)
                end
            end
        end
    end

    # The matrix that has to be p.s-d.
    M = Matrix{BosonOperator}(undef, length(gens), length(gens))
    for (i,g) in enumerate(gens)
        for (j,g′) in enumerate(gens)
            M[i,j] = g' * g
        end
    end

    # Algebraic identities. The list of returned matrices A is such that, at
    # any time, tracing A with the matrix of expectation values yields zero.
    # The remaining identity is that <1> = 1.
    A,b = let
        # Primal degrees of freedom
        m = Vector{Matrix{ComplexF64}}()
        for (k,op) in enumerate(basis)
            if op == Boson(0,0)
                continue
            end
            mat = zeros(ComplexF64, (length(gens),length(gens)))
            for i in 1:length(gens)
                for j in 1:length(gens)
                    if op in M[i,j]
                        mat[i,j] += M[i,j][op]
                    end
                end
            end
            push!(m, mat)
        end

        # Get orthogonal space (dual degrees of freedom)
        A = Vector{Matrix{ComplexF64}}()
        b = Vector{Float64}()
        for i in 1:(length(gens)-length(m))
            # Generate random Hermitian matrix.
            mat = randn(ComplexF64, (length(gens),length(gens)))
            mat = mat + mat'
            # Orthogonalize against A and m
            for a in Iterators.flatten([A,m])
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

    # Equations of motion.
    C, D = let
        C = Vector{Matrix{ComplexF64}}()
        D = Vector{Matrix{ComplexF64}}()
        for b in basis
            # Value
            # TODO
            # Derivative
            # TODO
        end
        C,D
    end

    # Construct the SDP.
    sdp = CompositeSDP(1,[1])

    sol = CONCAVE.IPM.solve(sdp; verbose=true)
end

function demo(::Val{:SpinRT})
    # Parameters.
    J = 1.

    # Construct operators.
    I,X,Y,Z = SpinAlgebra()

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

