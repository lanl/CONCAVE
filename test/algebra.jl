module TestAlgebras

using Test

using CONCAVE

@testset "Majorana" begin
    I,γ = MajoranaAlgebra()
    @assert I*γ ≈ γ
    @assert γ*γ ≈ I
    @assert 2*γ ≈ γ + γ
    @assert I+γ ≈ γ+I
    @assert I-γ ≆ γ-I
    @assert I-γ ≈ -(γ-I)
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
