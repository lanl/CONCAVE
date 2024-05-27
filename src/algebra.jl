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

export Operator
export Majorana, MajoranaAlgebra

abstract type Basis end

struct Operator{B<:Basis}
    terms::Dict{B,ComplexF64}

    function Operator{B}() where {B<:Basis}
        new(Dict{B,ComplexF64}())
    end

    function Operator{B}(b::B) where {B<:Basis}
        new(Dict{B,ComplexF64}(b => one(ComplexF64)))
    end
end

function Operator(b::B) where {B<:Basis}
    Operator{B}(b)
end

function ==(a::Operator{B}, b::Operator{B})::Bool where {B}
    return a.terms == b.terms
end

function isapprox(a::Operator{B}, b::Operator{B})::Bool where {B}
    for op in keys(a.terms) ∪ keys(b.terms)
        if !isapprox(a.terms[op], b.terms[op], atol=1e-12)
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

function zero(::Type{Operator{B}})::Operator{B} where {B}
    return Operator{B}()
end

function one(::Type{Operator{B}})::Operator{B} where {B}
    return Operator(one(B))
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
        r.terms[op] = -r.terms[op]
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
            r += a.terms[a′]*b.terms[b′] * (a′*b′)
        end
    end
    return r
end

function *(c::ComplexF64, a::Operator{B})::Operator{B} where {B<:Number}
    r = copy(a)
    for op in keys(r.terms)
        r.terms[op] = c*r.terms[op]
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
        r = r+adjoint(a.terms[op])*adjoint(op)
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

end
