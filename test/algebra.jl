module TestAlgebras

using Test

using CONCAVE

@testset "Majorana" begin
    I,γ = MajoranaAlgebra()
    @test I*γ ≈ γ
    @test γ*γ ≈ I
    @test 2*γ ≈ γ + γ
    @test I+γ ≈ γ+I
    @test !(I-γ ≈ γ-I)
    @test I-γ ≈ -(γ-I)
end

@testset "Pauli" begin
end

@testset "Fermion" begin
end

@testset "Boson" begin
end

@testset "Wick" begin
end

end
