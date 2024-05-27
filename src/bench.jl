using CONCAVE

using ArgParse

function main()
    args = let
        s = ArgParseSettings()
        @add_arg_table s begin
        end
        parse_args(s)
    end
end

main()

