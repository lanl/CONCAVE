#=
Algebras to implement:
* Majorana
* Dirac
* Pauli
* Oscillator
=#

module Algebras

import Base: +,-,*,^,adjoint
import Base: zero, one
import Base: copy, hash, ==, isapprox, isless, show
import Base: setindex!, getindex

export Operator
export Majorana, MajoranaAlgebra

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

function adjoint(γ::Majorana)::Majorana
    return copy(γ)
end

function *(γ1::Majorana, γ2::Majorana)::MajoranaOperator
    return Operator(Majorana(γ1.γ ⊻ γ2.γ))
end

end
