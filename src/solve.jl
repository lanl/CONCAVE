using ArgParse

using CONCAVE

demo(s::Symbol) = demo(Val(s))

function demo(::Val{:RT})
    # Parameters.
    ω = 1.
    λ = 0.1

    # For diagonalizing.
    p = CONCAVE.Hamiltonians.Oscillator(ω, λ)
    ham = CONCAVE.Hamiltonians.Hamiltonian(p)

    # Construct operators.
    I,a = BosonAlgebra()
    x = sqrt(1/(2*ω)) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)

    # Build the Hamiltonian.
    H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4

    # Generators of the SDP.
    gens = [I, x, p, x^2, p^2, x*p]

    # Construct the SDP.
    sdp = CompositeSDP(1,[1])

    # Solve the SDP
    sol = CONCAVE.IPM.solve(sdp; verbose=true)
    println(sol)
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

