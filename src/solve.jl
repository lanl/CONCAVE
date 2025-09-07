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
    L::Int
    g1::Float64
    g2::Float64
    μ::Float64
    a::Float64

    function NeutronMatterProgram(L::Int, g1::Float64, g2::Float64, μ::Float64, a::Float64; periodic=true)
        V = L^3

        function index(x,y,z)
            return (x-1)*L*L + (y-1)*L + z
        end

        # Construct algebra
        @algebra NeutronAlgebra begin
            I::Identity
            cu::Dirac[V]
            cd::Dirac[V]
        end

        # Construct Hamiltonian
        H = 0*I
        # Hopping, and neighbor interaction
        function hopping(i,j)
            hu = adjoint(cu[i])*cu[j] + adjoint(cu[j]) * cu[i]
            hd = adjoint(cd[i])*cd[j] + adjoint(cd[j]) * cd[i]
            # TODO coefficient
            return hu + hd
        end
        function nninteraction(i,j)
            ni = adjoint(cu[i])*cu[i] + adjoint(cd[i])*cd[i]
            nj = adjoint(cu[j])*cu[j] + adjoint(cd[j])*cd[j]
            # TODO coefficient
            return ni*nj
        end
        for x in 1:L, y in 1:L, z in 1:L
            xp = mod1(x+1,L)
            yp = mod1(y+1,L)
            zp = mod1(z+1,L)
            wrapx = xp < x
            wrapy = yp < y
            wrapz = zp < z
            i = index(x,y,z)
            if !wrapx || periodic
                j = index(xp,y,z)
                H += hopping(i,j)
                H += nninteraction(i,j)
            end
            if !wrapy || periodic
                j = index(x,yp,z)
                H += hopping(i,j)
                H += nninteraction(i,j)
            end
            if !wrapz || periodic
                j = index(x,y,zp)
                H += hopping(i,j)
                H += nninteraction(i,j)
            end
        end
        # Same-site interaction.
        for x in 1:L, y in 1:L, z in 1:L
            i = index(x,y,z)
            nu = adjoint(cu[i])*cu[i]
            nd = adjoint(cd[i])*cd[i]
            # TODO lattice spacing
            H += g1 * nu * nd
        end

        # Chemical potential
        N = 0*I
        for x in 1:L, y in 1:L, z in 1:L
            i = index(x,y,z)
            N += adjoint(cu[i])*cu[i]
            N += adjoint(cd[i])*cd[i]
        end
        H -= μ*N

        # Construct the psd matrices.
        M = [] # TODO

        # Identify constraint matrices.
        # TODO
        
        # Construct objective matrix.
        C = 0 # TODO

        new(L,g1,g2,μ,a,A,b,C)
    end
end

function size(p::NeutronMatterProgram)::Int
    # TODO
    return 1
end

function initial(p::NeutronMatterProgram)::Vector{Float64}
    return rand(Float64, size(p))
end

function objective!(g, p::NeutronMatterProgram, y::Vector{Float64})::Float64
    # TODO
    return 0.0
end

function constraints!(cb, p::NeutronMatterProgram, y::Vector{Float64})
    # TODO
    # cb(Λ, dΛ, hessian). We expect the hessian to be 0.
end

function demo(::Val{:NeutronMatter}, verbose::Bool)
    L::Int = 4
    g1::Float64 = 0.1
    g2::Float64 = 0.1
    a::Float64 = 0.5

    for μ in 0:0.1:1.0
        plo = NeutronMatterProgram(L, g1, g2, μ, a)
        phi = NeutronMatterProgram(L, g1, g2, μ, a)

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

