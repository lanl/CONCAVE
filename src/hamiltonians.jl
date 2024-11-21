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

struct TwoOscillators
    ω::Float64
    λ::Float64
end

function Hamiltonian(par::TwoOscillators)::Hamiltonian{TwoOscillators}
    ω,λ = par.ω,par.λ
    N = 40
    I1 = zeros(ComplexF64, (N,N)) + I
    c = zeros(ComplexF64, (N,N))
    for i in 1:(N-1)
        c[i,i+1] = sqrt(ω * i)
    end
    a = kron(c,I1)
    b = kron(I1,c)

    x = 1/sqrt(2*ω) * (a + a')
    p = 1im * sqrt(ω/2) * (a' - a)
    y = 1/sqrt(2*ω) * (b + b')
    q = 1im * sqrt(ω/2) * (b' - b)

    H = 0.5*(p^2 + q^2) + 0.5*ω^2*(x^2 + y^2) + 0.5*(x-y)^2 + 0.25*λ*(x^4 + y^4)

    F = eigen(Hermitian(H))
    return Hamiltonian{TwoOscillators}(Dict("I"=>I1,"x"=>x,"y"=>y,"x²"=>x^2,"p"=>p,"q"=>q),H,F)
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
                a = kron(a, 0.5*(pauli_X + 1im * pauli_Y))
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
    end
    # Interaction
    for x in 1:p.L
        H .+= p.U * c[1,x]' * c[1,x] * c[2,x]' * c[2,x]
    end

    # Total number
    N = zeros(ComplexF64, (4^p.L,4^p.L))
    for s in 1:2, x in 1:p.L
        N += c[s,x]' * c[s,x]
    end

    # Average position
    x̂ = zeros(ComplexF64, (D,D))
    for s in 1:2, x in 1:p.L
        x̂ += cos(2 * π * x / p.L) * c[s,x]' * c[s,x]
    end

    F = eigen(Hermitian(H))
    return Hamiltonian{FermiHubbardChain}(Dict("N"=>N,"x"=>x̂),H,F)
end

function build_state(fn, p::FermiHubbardChain)::Vector{ComplexF64}
    ψ = zeros(ComplexF64, 4^p.L)
    for (i,nu) in enumerate(Iterators.product(ntuple(_->(false,true),p.L)))
        for (j,nd) in enumerate(Iterators.product(ntuple(_->(false,true),p.L)))
            ψ[(i-1)*2^p.L + j] = fn(nu,nd)
        end
    end
    ψ /= norm(ψ)
    return ψ
end

function basis_state(fn, p::FermiHubbardChain)::Vector{ComplexF64}
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
                a = kron(a, 0.5 * (pauli_X + 1im * pauli_Y))
            else
                a = kron(a, pauli_I)
            end
        end
        c[s,x] = a
    end

    N = zeros(ComplexF64, (4^p.L,4^p.L))
    for s in 1:2, x in 1:p.L
        N += c[s,x]' * c[s,x]
    end
    F = eigen(Hermitian(N))

    ψ = F.vectors[:,1]
    for s in 1:2, x in 1:p.L
        if fn(s,x)
            ψ = c[s,x]' * ψ
        end
    end
    return ψ
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
