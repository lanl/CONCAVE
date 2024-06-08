module TestIPM

using Test

using CONCAVE

@testset verbose=true "Examples" begin
    @testset "Two dimensions" begin
        sdp = CompositeSDP(2, [2])
        sdp.h[2] = -1

        CONCAVE.IPM.solve(sdp)
    end

    @testset "Single spin" begin
    end

    @testset "Harmonic oscillator" begin
    end
end

@testset "Random" begin
end

end
