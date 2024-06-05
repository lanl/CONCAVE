using ArgParse

using CONCAVE

demo(s::Symbol) = demo(Val(s))

function demo(::Val{:RT})
    I,c = BosonAlgebra()
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

