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

function demo(::Val{:Neutrons}, verbose::Bool)
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

