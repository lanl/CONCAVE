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
