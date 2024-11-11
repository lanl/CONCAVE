module Hamiltonians

using LinearAlgebra

struct Hamiltonian{P}
    op::Dict{String,Matrix{ComplexF64}}
    H::Matrix{ComplexF64}
    F::Eigen
end

function evolution(h::Hamiltonian, t::Float64)::Matrix{ComplexF64}
    return h.F.vectors * diagm(exp.(-1im * t * h.F.values)) * h.F.vectors'
end

struct Oscillator
    ω::Float64
    λ::Float64
end

function Hamiltonian(par::Oscillator)::Hamiltonian{Oscillator}
    ω, λ = par.ω, par.λ
    N = 100
    I = zeros(ComplexF64, (N,N))
    for i in 1:N
        I[i,i] = 1.0
    end
    a = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        a[i,i+1] = sqrt(ω * i)
    end
    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    H = 0.5*p^2 + 0.5 * ω^2 * x^2 + 0.25 * λ * x^4
    F = eigen(Hermitian(H))
    return Hamiltonian{Oscillator}(Dict("I"=>I,"x"=>x,"p"=>p,"a"=>a),H,F)
end

struct FermiHubbardChain
    L::Int
    t::Float64
    U::Float64
end

function Hamiltonian(p::FermiHubbardChain)::Hamiltonian{FermiHubbardChain}
    # Pauli operators
    pauli_I::Matrix{ComplexF64} = zeros(ComplexF64, (2,2)) + I
    pauli_X::Matrix{ComplexF64} = [0 1; 1 0]
    pauli_Y::Matrix{ComplexF64} = [0 -1im; 1im 0]
    pauli_Z::Matrix{ComplexF64} = [1 0; 0 -1]

    D::Int = 4^p.L

    # Fermion annihilation operators
    c = Matrix{Matrix{ComplexF64}}(undef, (2,p.L))
    for s in 1:2, x in 1:p.L
        a = zeros(ComplexF64, (1,1)) .+ 1
        for s′ in 1:2, x′ in 1:p.L
            if (s,x) < (s′,x′)
                a = kron(a, pauli_Z)
            elseif (s,x) == (s′,x′)
                a = kron(a, pauli_X + 1im * pauli_Y)
            else
                a = kron(a, pauli_I)
            end
        end
        c[s,x] = a
    end

    # Construct Hamiltonian.
    H = zeros(ComplexF64, (D,D))
    # Hopping
    for s in 1:2, x in 1:p.L
        x′ = mod1(x+1,p.L)
        H .+= -p.t * (c[s,x]' * c[s,x′] + c[s,x′]' * c[s,x])
        println(size(H))
    end
    # Interaction
    for x in 1:p.L
        H .+= p.U * c[1,x]' * c[1,x] * c[2,x]' * c[2,x]
    end

    F = eigen(Hermitian(H))
    return Hamiltonian{FermiHubbardChain}(Dict(),H,F)
end

function basis_state(f, p::FermiHubbardChain)
    # TODO
end

struct LatticeNeutron
end

function Hamiltonian(p::LatticeNeutron)::Hamiltonian{LatticeNeutron}
end

struct LatticeNeutrons
end

function Hamiltonian(p::LatticeNeutrons)::Hamiltonian{LatticeNeutrons}
end

end
