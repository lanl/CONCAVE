module TestAlgebras

using Test

using Random

using CONCAVE

@testset "SinglePauli" begin
    @algebra SinglePauli begin
        σ::Pauli
    end
    @test σ[1] ≈ σ[1]
    @test !(σ[1] ≈ σ[2])
    @test adjoint(σ[1]) ≈ σ[1]
    @test σ[1] * σ[1] ≈ σ[1] * σ[1]
    @test σ[1] * σ[1] ≈ σ[2] * σ[2]
    @test σ[1] * σ[2] ≈ 1im * σ[3]
    @test !(σ[1] * σ[2] ≈ -1im * σ[3])
end

@testset "PauliAlgebra" begin
    @algebra PauliAlgebra begin
        σ::Pauli[3]
    end
    for (i,j) in zip(1:3,1:3)
        @test σ[1][i] * σ[2][j] ≈ σ[2][j] * σ[1][i]
    end
end

@testset "SingleDirac" begin
    @algebra SingleDirac begin
        a::Dirac
    end
    @test a ≈ a
    @test !(adjoint(a) ≈ a)
    @test adjoint(a) ≈ adjoint(a)
end

@testset "TwoDiracs" begin
    @algebra TwoDiracs begin
        I::Identity
        a::Dirac
        b::Dirac
    end
    @test a * b ≈ -b * a
    @test !(a * b ≈ b * a)
    @test a * a ≈ 0*a
    @test !(adjoint(a) * a ≈ a * adjoint(a))
    @test I - adjoint(a) * a ≈ a * adjoint(a)
end

@testset "DiracAlgebra" begin
    @algebra DiracAlgebra begin
        a::Dirac[8]
    end
    @test a[1] * a[2] ≈ - a[2] * a[1]
    @test a[1] * a[2] * adjoint(a[3]) ≈ - adjoint(a[3]) * a[2] * a[1]
    @test a[1] * a[2] * adjoint(a[3]) * a[4] ≈ a[4] * adjoint(a[3]) * a[2] * a[1]
    @test adjoint(a[1])*a[1] * a[2] ≈ a[2] * adjoint(a[1]) * a[1]
end

@testset "SingleMajorana" begin
    @algebra SingleMajorana begin
        γ::Majorana
    end
    @test !(γ ≈ γ*γ)
    @test γ*γ ≈ γ*γ*γ
end

@testset "MajoranaAlgebra" begin
    @algebra MajoranaAlgebra begin
        γ::Majorana[8]
    end
    for i in 1:8
        for j in 1:8
            @test γ[i] * γ[j] ≈ -γ[j] * γ[i]
        end
    end
    for i in 1:8
        for j in 1:8
            for k in 1:8
                @test γ[i] * (γ[j] * γ[k]) ≈ (γ[i] * γ[j]) * γ[k]
            end
        end
    end
end

@testset "SingleBose" begin
    @algebra SingleBose begin
        I::Identity
        c::Bose
    end
    @test I*c ≈ c*I
    @test c*c ≈ c*c
    @test adjoint(c*c) ≈ adjoint(c) * adjoint(c)
    @test c*adjoint(c) ≈ I + adjoint(c) * c
    @test c*c*adjoint(c) ≈ c + c*adjoint(c)*c
    @test c*adjoint(c)*adjoint(c) ≈ adjoint(c*c*adjoint(c))
end

@testset "BoseAlgebra" begin
    @algebra BoseAlgebra begin
        I::Identity
        c::Bose[8]
    end
end

@testset "BigAlgebra" begin
    K::Int = 6
    @algebra BigAlgebra begin
        σ::Pauli[K]
        a::Dirac[K]
        γ::Majorana[K]
        c::Bose[K]
    end
    function random()::Operator
        op = σ[1][1]
        for k in 1:K
            i = rand(0:3)
        end
        return op
    end
end

end
