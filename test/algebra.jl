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
    cdag = adjoint(c)
    @test c*c ≈ 0*I
    @test c*I ≈ c
    @test cdag*I ≈ cdag
    @test !(c ≈ cdag)
    @test I*c ≈ c
    @test I*cdag ≈ cdag
    @test cdag*cdag ≈ 0*I
    @test cdag*c + c*cdag ≈ I
    @test cdag*c * cdag*c ≈ cdag*c
end

@testset "Boson" begin
    I,a = BosonAlgebra()
    adag = adjoint(a)
    @test a*I ≈ a
    @test I*a ≈ a
    @test adag*I ≈ adag
    @test I*adag ≈ adag
    @test a*adag - adag*a ≈ I
    @test a*adag*a*adag ≈ -a*adag + a*a*adag*adag
end

@testset "Wick" begin
end

end
