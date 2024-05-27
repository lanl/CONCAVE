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
    I,X,Y,Z = PauliAlgebra()
    @test I*I ≈ I
    @test I*X ≈ X
    @test I*Y ≈ Y
    @test I*Z ≈ Z
    @test X*Y ≈ 1im*Z
    @test Y*Z ≈ -Z*Y
    @test Y*Y ≈ I
end

@testset "Fermion" begin
    I,c = FermionAlgebra()
end

@testset "Boson" begin
    I,a = BosonAlgebra()
end

@testset "Wick" begin
end

end
