#=
Algebras to implement:
* Majorana
* Dirac
* Pauli
* Oscillator
=#

module Algebras

import Base: +,-,*,/,^,adjoint
import Base: zero, one, isone
import Base: copy, hash, ==, isapprox, isless, show
import Base: setindex!, getindex

export Operator
export Majorana, MajoranaAlgebra
export Pauli, PauliAlgebra
export Fermion, FermionAlgebra
export Boson, BosonAlgebra
export Wick, WickAlgebra

abstract type Basis end

struct Operator{B<:Basis}
    terms::Dict{B,ComplexF64}
end

function Operator{B}() where {B<:Basis}
    Operator{B}(Dict{B,ComplexF64}())
end

function Operator{B}(b::B) where {B<:Basis}
    Operator{B}(Dict{B,ComplexF64}(b => one(ComplexF64)))
end

function Operator(b::B) where {B<:Basis}
    Operator{B}(b)
end

function ==(a::Operator{B}, b::Operator{B})::Bool where {B}
    return a.terms == b.terms
end

function isapprox(a::Operator{B}, b::Operator{B})::Bool where {B}
    for op in keys(a.terms) ∪ keys(b.terms)
        if !isapprox(a[op], b[op], atol=1e-12)
            return false
        end
    end
    return true
end

function hash(a::Operator{B}, h::UInt)::UInt where {B}
    return hash(a.terms, h)
end

function copy(a::Operator{B})::Operator{B} where {B}
    return Operator(copy(a.terms))
end

function getindex(a::Operator{B}, b::B)::ComplexF64 where {B}
    if b in keys(a.terms)
        return a.terms[b]
    else
        return zero(ComplexF64)
    end
end

function setindex!(a::Operator{B}, c::ComplexF64, b::B)::ComplexF64 where {B}
    a.terms[b] = c
    return c
end

function zero(::Type{Operator{B}})::Operator{B} where {B}
    return Operator{B}()
end

function one(::Type{Operator{B}})::Operator{B} where {B}
    return Operator(one(B))
end

function add!(a::Operator{B}, b::Operator{B}, c::ComplexF64=one(ComplexF64))::Operator{B} where {B}
    for op in keys(b.terms)
        a[op] = a[op] + c*b[op]
    end
    return a
end

function +(a::Operator{B}, b::Operator{B})::Operator{B} where {B}
    r = zero(Operator{B})
    add!(r, a)
    add!(r, b)
    return r
end

function -(a::Operator{B})::Operator{B} where {B}
    r = copy(a)
    for op in keys(r.terms)
        r[op] = -r[op]
    end
    return r
end

function -(a::Operator{B}, b::Operator{B})::Operator{B} where {B}
    return a + (-b)
end

function *(a::Operator{B}, b::Operator{B})::Operator{B} where {B}
    r = zero(Operator{B})
    for a′ in keys(a.terms)
        for b′ in keys(b.terms)
            r += a[a′]*b[b′] * (a′*b′)
        end
    end
    return r
end

function *(c, a::Operator{B})::Operator{B} where {B}
    r = copy(a)
    for op in keys(r.terms)
        r[op] = c*r[op]
    end
    return r
end

function *(a::Operator{B}, c::ComplexF64)::Operator{B} where {B}
    return c*a
end

function /(a::Operator, c)
    return (1/c)*a
end

function ^(a::Operator{B}, n::Int)::Operator{B} where {B}
    if n == 0
        return one(Operator{B})
    elseif n == 1
        return a
    else
        return a * (a^(n-1))
    end
end

function adjoint(a::Operator{B})::Operator{B} where {B}
    r = zero(Operator{B})
    for op in keys(a.terms)
        r = r+adjoint(a[op])*adjoint(op)
    end
    return r
end

function show(io::IO, ::MIME"text/plain", op::Operator{B}) where {B}
    op = trim(op)
    if isempty(op.terms)
        println(io, " 0")
    end
    for (b,c) in op.terms
        print(io, "  + ")
        print(io, c)
        print(io, " ")
        print(io, b)
        println(io)
    end
end

struct Majorana <: Basis
    γ::Bool
end

MajoranaOperator = Operator{Majorana}

function MajoranaAlgebra()
    return Operator(Majorana(false)), Operator(Majorana(true))
end

function ==(γ1::Majorana, γ2::Majorana)::Bool
    return γ1.γ == γ2.γ
end

function hash(γ::Majorana, h::UInt)::UInt
    return hash(γ.γ, h)
end

function copy(γ::Majorana)::Majorana
    return Majorana(γ.γ)
end

function one(::Type{Majorana})::Majorana
    return Majorana(false)
end

function isone(γ::Majorana)::Bool
    return γ.γ == false
end

function adjoint(γ::Majorana)::MajoranaOperator
    return Operator(γ)
end

function *(γ1::Majorana, γ2::Majorana)::MajoranaOperator
    return Operator(Majorana(γ1.γ ⊻ γ2.γ))
end

struct Pauli <: Basis
    p::Char
end

PauliOperator = Operator{Pauli}

function PauliAlgebra()
    return Operator(Pauli('I')), Operator(Pauli('X')), Operator(Pauli('Y')), Operator(Pauli('Z'))
end

function ==(a::Pauli, b::Pauli)
    return a.p == b.p
end

function hash(a::Pauli, h::UInt)::UInt
    return hash(a.p, h)
end

function copy(a::Pauli)::Pauli
    return Pauli(a.p)
end

function one(::Type{Pauli})
    return Pauli('I')
end

function isone(a::Pauli)::Bool
    return a.p == 'I'
end

function adjoint(a::Pauli)::PauliOperator
    return Operator(a)
end

function *(a::Pauli, b::Pauli)::PauliOperator
    if a.p == 'I'
        return Operator(b)
    elseif b.p == 'I'
        return Operator(a)
    end
    if a.p == b.p
        return Operator(Pauli('I'))
    end
    if a.p == 'X' && b.p == 'Y'
        return 1im * Operator(Pauli('Z'))
    elseif a.p == 'Y' && b.p == 'Z'
        return 1im * Operator(Pauli('X'))
    elseif a.p == 'Z' && b.p == 'X'
        return 1im * Operator(Pauli('Y'))
    end
    if a.p == 'Y' && b.p == 'X'
        return -1im * Operator(Pauli('Z'))
    elseif a.p == 'Z' && b.p == 'Y'
        return -1im * Operator(Pauli('X'))
    elseif a.p == 'X' && b.p == 'Z'
        return -1im * Operator(Pauli('Y'))
    end
end

struct Fermion <: Basis
    cr::Bool
    an::Bool
end

FermionOperator = Operator{Fermion}

function FermionAlgebra()
    return Operator(Fermion(false,false)), Operator(Fermion(false,true))
end

function ==(a::Fermion, b::Fermion)
    return a.cr == b.cr && a.an == b.an
end

function hash(a::Fermion, h::UInt)
    return hash((a.cr, a.an), h)
end

function copy(a::Fermion)::Fermion
    return Fermion(a.cr, a.an)
end

function one(::Type{Fermion})
    return Fermion(0,0)
end

function isone(a::Fermion)::Bool
    return a.an == false && a.cr == false
end

function adjoint(a::Fermion)::FermionOperator
    return Operator(Fermion(a.an, a.cr))
end

function *(a::Fermion, b::Fermion)::FermionOperator
    if a.an == false || b.cr == false
        if (a.an && b.an) || (a.cr && b.cr)
            return zero(FermionOperator)
        end
        cr = a.cr || b.cr
        an = a.an || b.an
        return Operator(Fermion(cr,an))
    elseif a.cr == false && b.an == false
        return one(FermionOperator) - Operator(Fermion(true,true))
    else
        return Operator(Fermion(a.cr,b.an))
    end
end

function isfermionic(a::Fermion)::Bool
    return a.cr ⊻ a.an
end

struct Boson <: Basis
    cr::Int
    an::Int
end

BosonOperator = Operator{Boson}

function BosonAlgebra()
    return Operator(Boson(0,0)), Operator(Boson(0,1))
end

function ==(a::Boson, b::Boson)
    return a.cr == b.cr && a.an == b.an
end

function hash(a::Boson, h::UInt)
    return hash((a.cr, a.an), h)
end

function copy(a::Boson)::Boson
    return Boson(a.cr, a.an)
end

function one(::Type{Boson})
    return Boson(0,0)
end

function isone(a::Boson)::Bool
    return a.cr == 0 && a.an == 0
end

function adjoint(a::Boson)::BosonOperator
    return Operator(Boson(a.an, a.cr))
end

function *(a::Boson, b::Boson)::BosonOperator
    # Base case:
    if a.an == 0
        return Operator(Boson(a.cr+b.cr,b.an))
    end
    if b.cr == 0
        return Operator(Boson(a.cr,a.an+b.an))
    end
    # Recurse
    opl = Operator(Boson(a.cr, a.an-1))
    opr = Operator(Boson(b.cr-1, b.an))
    return opl*opr + opl*Operator(Boson(1,1))*opr
end

struct Wick <: Basis
    b::Dict{String,Boson}
    f::Dict{String,Fermion}
end

WickOperator = Operator{Wick}

function WickAlgebra()
    I = WickOperator(one(Wick))

    function fan(n::String)::WickOperator
        r = one(Wick)
        r.f[n] = Fermion(false,true)
        return WickOperator(r)
    end

    function ban(n::String)::WickOperator
        r = one(Wick)
        r.b[n] = Boson(0,1)
        return WickOperator(r)
    end

    return I, ban, fan
end

function ==(a::Wick, b::Wick)
    return a.b == b.b && a.f == b.f
end

function hash(a::Wick, h::UInt)::UInt
    return hash((a.b,a.f), h)
end

function copy(a::Wick)::Wick
    return Wick(copy(a.b), copy(a.f))
end

function one(::Type{Wick})
    return Wick(Dict{String,Boson}(), Dict{String,Fermion}())
end

function isone(a::Wick)::Bool
    return isempty(a.b) && isempty(a.f)
end

function adjoint(a::Wick)::WickOperator
    r = copy(a)
    # Adjoint everything, and count fermionic operators.
    nf = 0
    for (n,op) in r.b
        r.b[n] = Boson(op.an, op.cr)
    end
    for (n,op) in r.f
        r.f[n] = Fermion(op.an, op.cr)
        nf += isfermionic(r.f[n])
    end
    nswaps = round(Int, (nf * (nf-1)/2))
    return (-1)^nswaps * WickOperator(r)
end

function *(a::Wick, b::Wick)::WickOperator
    # Operate recursively. The base case: check if either is the identity.
    if isone(a)
        return WickOperator(copy(b))
    elseif isone(b)
        return WickOperator(copy(a))
    end
    # There are both fermionic and bosonic modes in general. If there are
    # fermionic modes we'll recurse on them first.
    if isempty(a.f) && isempty(b.f)
        # There are no fermionic modes.
        mode = min(minimum(keys(a.b)), minimum(keys(b.b)))
        # Take the product of this mode.
        mp = one(BosonOperator)
        if mode in keys(a.b)
            mp = mp * BosonOperator(a.b[mode])
        end
        if mode in keys(b.b)
            mp = mp * BosonOperator(b.b[mode])
        end
        # Take the product of what remains.
        a′ = copy(a)
        b′ = copy(b)
        delete!(a′.b, mode)
        delete!(b′.b, mode)
        p = a′*b′
        # Merge.
        r = zero(WickOperator)
        for (mop,mc) in mp.terms
            for (wp,wc) in p.terms
                w = copy(wp)
                if !isone(mop)
                    w.b[mode] = mop
                end
                r += mc*wc*WickOperator(w)
            end
        end
        return r
    else
        # Find the first fermionic mode.
        mode = min(minimum(keys(a.f)), minimum(keys(b.f)))
        # Take the product of this mode.
        mp = one(FermionOperator)
        if mode in keys(a.f)
            mp = mp * FermionOperator(a.f[mode])
        end
        if mode in keys(b.f)
            mp = mp * FermionOperator(b.f[mode])
        end
        # Take the product of what remains.
        a′ = copy(a)
        b′ = copy(b)
        delete!(a′.f, mode)
        delete!(b′.f, mode)
        p = a′*b′
        # Compute the sign. This sign comes from needing to commute `mode` from b,
        # through all fermionic operators in a′.
        naf = 0
        for (m,aop) in a′.f
            naf += isfermionic(aop)
        end
        if mode in keys(b.f) && isfermionic(b.f[mode])
            sgn = (-1.)^(naf)
        else
            sgn = 1.
        end
        # Merge.
        r = zero(WickOperator)
        for (mop,mc) in mp.terms
            for (wp,wc) in p.terms
                w = copy(wp)
                if !isone(mop)
                    w.f[mode] = mop
                end
                r += mc*wc*WickOperator(w)
            end
        end
        return sgn * r
    end
end

end
