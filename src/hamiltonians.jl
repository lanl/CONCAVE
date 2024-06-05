module Hamiltonians

using LinearAlgebra

struct Hamiltonian{P}
    H::Matrix{ComplexF64}
    F::Eigen
end

struct Oscillator
    ω::Float64
    λ::Float64
end

function Hamiltonian(par::Oscillator)::Hamiltonian{Oscillator}
    ω, λ = par.ω, par.λ
    N = 200
    a = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        a[i,i+1] = sqrt(ω * i)
    end
    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    H = 0.5*p^2 + 0.5 * ω^2 * x^2 + 0.25 * λ * x^4
    F = eigen(Hermitian(H))
    return Hamiltonian{Oscillator}(H,F)
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
