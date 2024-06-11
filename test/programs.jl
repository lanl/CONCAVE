module TestPrograms

using LinearAlgebra
using Test

using CONCAVE

@testset verbose=true "CompositeSDP" begin

    @testset "barrier" begin
        sdp = CompositeSDP(2, [2])
        sdp.h[2] = -1
        sdp.M₀[1][2,2] = 1
        sdp.M[1][1,1,1] = 1
        sdp.M[1][2,1,2] = 1
        sdp.M[1][1,2,2] = 1

        h = 1e-4
        y = [3., 1.2]
        r, ∇ = CONCAVE.Programs.barrier(sdp, y)
        y′ = y - h * ∇
        r′, ∇′ = CONCAVE.Programs.barrier(sdp, y′)
        @test abs((r-r′) - h * (∇ ⋅ ∇)) < 100*h^2
    end

end

end
