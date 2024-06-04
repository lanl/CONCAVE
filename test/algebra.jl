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

@testset verbose=true "Wick" begin
    @testset "Bosonic" begin
        I,ban,fan = WickAlgebra()
        a,b,c = ban("a"), ban("b"), ban("c")
        ops = [I,a,b,c,a',b',c',a*a',a*b']
        @testset "Commutativity" begin
            for op1 in ops
                for op2 in ops
                end
            end
        end

        @testset "Associativity" begin
            opa = [I, a, a', a'*a]
            opb = [I, b, b', b'*b]
            ops = []
            for a′ in opa
                for b′ in opb
                    push!(ops, a′*b′)
                end
            end

            for op1 in ops
                for op2 in ops
                    for op3 in ops
                        @test op1*(op2*op3) ≈ (op1*op2)*op3
                    end
                end
            end
        end
    end

    @testset "Fermionic" begin
        I,ban,fan = WickAlgebra()
        z = 0*I
        a,b,c = fan("a"), fan("b"), fan("c")
        @test a == a
        @test a' == a'
        @test a ≠ a'
        @test (a*b)' == b'*a'
        @test (a*b)' ≠ a'*b'
        @test (a*b*c)' == c' * b' * a'

        @test a*I == a
        @test I*a == a
        @test a*b == -b*a
        @test a*b' == -b'*a

        @test a*a ≈ z
        @test (a'*a)*(a'*a) ≈ a'*a

        @test I*I == I
        @test (a*b)*(I*I) == ((a*b)*I)*I
        @test (a*b)*I == a*b

        @test (a*b*c)' ≈ - a' * b' * c'
        @testset "Associativity" begin
            opa = [I, a, a', a'*a]
            opb = [I, b, b', b'*b]
            opc = [I, c, c', c'*c]
            ops = []
            for a′ in opa
                for b′ in opb
                    push!(ops, a′*b′)
                end
            end

            for op1 in ops
                for op2 in ops
                    for op3 in ops
                        @test op1*(op2*op3) ≈ (op1*op2)*op3
                    end
                end
            end
        end
    end

    @testset "Mixed" begin
        I,ban,fan = WickAlgebra()
        @test adjoint(I) ≈ I
        @test I*I ≈ I
        @test I*ban("a") ≈ ban("a")
        @test fan("a")*I ≈ fan("a")
    end
end

end
