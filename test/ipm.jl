module TestIPM

using Test

using CONCAVE

@testset verbose=true "Examples" begin
    @testset "Two dimensions" begin
        sdp = CompositeSDP(1, [2])
        sdp.h[1] = -1
        sdp.M₀[1][1,1] = 1
        sdp.M₀[1][2,2] = 1
        sdp.M[1][2,1,1] = 1
        sdp.M[1][1,2,1] = 1

        r, y = CONCAVE.IPM.solve(sdp)
        @test abs(r+1) < 1e-4
    end

    @testset "Single spin" begin
    end

    @testset "Harmonic oscillator" begin
    end
end

@testset "Random" begin
end

end
