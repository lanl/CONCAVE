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

struct NeutronMatterProgram <: ConvexProgram
end

function demo(::Val{:NeutronMatter}, verbose::Bool)
    L::Int = 4
    V = L^3
    @algebra NeutronAlgebra begin
        c::Dirac[V]
    end

    for μ in 0:0.1:1.0
        plo = NeutronMatterProgram()
        phi = NeutronMatterProgram()

        lo, ylo = CONCAVE.IPM.solve(plo; verbose=verbose)
        hi, yhi = CONCAVE.IPM.solve(phi; verbose=verbose)

        if -lo > hi
            println(stderr, "WARNING: primal proved infeasible")
        end

        println("$μ $(-lo) $hi")
        flush(stdout)
    end
end

function demo(::Val{:Hubbard}, verbose::Bool)
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

    function action()
        if !isnothing(args["demo"])
            demo(args["demo"]; verbose=args["verbose"])
            return
        end
    end

    if args["profile"]
        @profile action()
        open("prof-flat", "w") do f
            Profile.print(f, format=:flat, sortedby=:count)
        end
        open("prof-tree", "w") do f
            Profile.print(f, noisefloor=2.0)
        end
    else
        action()
    end
end

main()

