using ArgParse

using CONCAVE

demo(s::Symbol) = demo(Val(s))

function demo(::Val{:RT})
    # Parameters.
    ω = 1.
    λ = 0.1

    # Construct operators.
    I,a = BosonAlgebra()
    x = sqrt(1/(2*ω)) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)

    # Build the Hamiltonian.
    H = p^2 / 2 + ω^2 * x^2 / 2 + λ * x^4 / 4
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
    end
end

main()

