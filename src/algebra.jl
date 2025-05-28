#!/usr/bin/env julia

#=
USAGE

In principle, an algebra is defined by specifying a finite number of
generators, and rules for their commutation (and possibly conjugation).

The principle is fairly abstract; the practice is less abstract. Each mode may
be labelled with something resembling a type, which is typically one of
{pauli,bose,dirac,majorana}. The commutation and conjugation rules are
determined by that type.

With that in mind, the notation to construct an algebra is:

    @algebra Name begin
        I::Identity
        σ::Pauli[5]
        γ::Majorana[3]
        a::Dirac
        c::Bose
    end

Hopefully this is self-explanatory. After this, algebraic manipulations may be
performed in the obvious way:
my_product = σ[2] * σ[4] + 0.3*γ

=#

#=
DESIGN NOTES

Holding the API fixed (which already makes a couple usability compromises for
the sake of efficiency), the next requirement is speed and scalability.
=#

#=
TODO

 * More comprehensive tests

 * Type information in BasisOperator?

 * Automatic trimming of vanishing coefficients

=#

import Base: +,-,*,/,^,adjoint
import Base: zero, one, isone
import Base: copy, hash, isequal, isapprox, isless, show
import Base: setindex!, getindex, in, iterate

using Random

abstract type Mode
end

@enum Pauli σI σX σY σZ

struct PauliMode <: Mode
    pauli::Pauli
end

struct DiracMode <: Mode
    cr::Bool
    an::Bool
end

struct MajoranaMode <: Mode
    n::Bool
end

struct BoseMode <: Mode
    cr::Int
    an::Int
end

one(::Type{PauliMode}) = PauliMode(σI)
one(::Type{DiracMode}) = DiracMode(0,0)
one(::Type{MajoranaMode}) = MajoranaMode(false)
one(::Type{BoseMode}) = BoseMode(0,0)

function show(io::IO, m::PauliMode)
    print(io, m.pauli)
end

function show(io::IO, m::DiracMode)
    if !(m.cr || m.an)
        print(io, "1")
    end
    if m.cr
        print(io, "a⁺")
    end
    if m.an
        print(io, "a")
    end
end

function show(io::IO, m::MajoranaMode)
    if m.n
        print(io, "γ")
    else
        print(io, "1")
    end
end

function show(io::IO, m::BoseMode)
    if m.cr == 0 && m.an == 0
        print(io, "1")
        return
    end
    for i in 1:m.cr
        print(io, "c⁺")
    end
    for i in 1:m.an
        print(io, "c")
    end
end

struct BasisOperator
    paulis::Vector{PauliMode}
    diracs::Vector{DiracMode}
    majoranas::Vector{MajoranaMode}
    boses::Vector{BoseMode}

    function BasisOperator(npauli::Int, ndirac::Int, nmajorana::Int, nbose::Int)
        paulis = [one(PauliMode) for k in 1:npauli]
        diracs = [one(DiracMode) for k in 1:ndirac]
        majoranas = [one(MajoranaMode) for k in 1:nmajorana]
        boses = [one(BoseMode) for k in 1:nbose]
        new(paulis,diracs,majoranas,boses)
    end
end

function copy(o::BasisOperator)
    r = BasisOperator(length(o.paulis), length(o.diracs), length(o.majoranas), length(o.boses))
    for k in 1:length(o.paulis)
        r.paulis[k] = o.paulis[k]
    end
    for k in 1:length(o.diracs)
        r.diracs[k] = o.diracs[k]
    end
    for k in 1:length(o.majoranas)
        r.majoranas[k] = o.majoranas[k]
    end
    for k in 1:length(o.boses)
        r.boses[k] = o.boses[k]
    end
    return r
end

function isequal(a::BasisOperator, b::BasisOperator)::Bool
    if length(a.paulis) != length(b.paulis)
        return false
    end
    if length(a.diracs) != length(b.diracs)
        return false
    end
    if length(a.majoranas) != length(b.majoranas)
        return false
    end
    if length(a.boses) != length(b.boses)
        return false
    end
    for (ma,mb) in zip(a.paulis,b.paulis)
        if ma != mb
            return false
        end
    end
    for (ma,mb) in zip(a.diracs,b.diracs)
        if ma != mb
            return false
        end
    end
    for (ma,mb) in zip(a.majoranas,b.majoranas)
        if ma != mb
            return false
        end
    end
    for (ma,mb) in zip(a.boses,b.boses)
        if ma != mb
            return false
        end
    end
    return true
end

function hash(a::BasisOperator, h::UInt=0x012456789)::UInt
    for m ∈ a.paulis
        h = hash(m, h)
    end
    for m ∈ a.diracs
        h = hash(m, h)
    end
    for m ∈ a.majoranas
        h = hash(m, h)
    end
    for m ∈ a.boses
        h = hash(m, h)
    end
    return h
end

struct Operator
    terms::Dict{BasisOperator,ComplexF64}

    function Operator()
        return new(Dict{BasisOperator,ComplexF64}())
    end

    function Operator(b::BasisOperator)
        return new(Dict{BasisOperator,ComplexF64}(b => 1.0))
    end
end

function show(io::IO, op::Operator)
    beginning = true
    for (b,c) ∈ op
        if !beginning
            print("+")
        end
        print("($c) * {$b}")
        beginning = false
    end
end

function copy(a::Operator)
    r = Operator()
    for (b,c) ∈ a
        r[b] = c
    end
    return r
end

function isapprox(a::Operator, b::Operator)
    for (x,c) ∈ a
        if !(b[x] ≈ c)
            return false
        end
    end
    for (x,c) ∈ b
        if !(a[x] ≈ c)
            return false
        end
    end
    return true
end

function add!(a::Operator, b::Operator, c::T=1) where {T <: Number}
    for (term,coef) ∈ b
        a[term] = a[term] + c*coef
    end
end

function scale!(a::Operator, c::ComplexF64)
    for (term,coef) ∈ a
        a[term] = c*coef
    end
end

function +(a::Operator, b::Operator)
    r = copy(a)
    add!(r, b)
    return r
end

function -(a::Operator, b::Operator)
    r = copy(a)
    add!(r, b, -1)
    return r
end

function *(op1::Operator, op2::Operator)::Operator
    r = Operator()
    for (b1,c1) in op1, (b2,c2) in op2
        bmul(b1,b2) do b,c
            r[b] += c1*c2*c
        end
    end
    return r
end

function *(c::T, op::Operator)::Operator where {T <: Number}
    r = copy(op)
    for (b,c′) in op
        r[b] = c*c′
    end
    return r
end

function -(op::Operator)::Operator
    return -1 * op
end

function adjoint(op::Operator)::Operator
    r = Operator()
    for (b,c) in op
        badjoint(b) do b′,c′
            r[b′] += c*c′
        end
    end
    return r
end

function getindex(a::Operator, b::BasisOperator)::ComplexF64
    if b in a
        return a.terms[b]
    else
        return zero(ComplexF64)
    end
end

function setindex!(a::Operator, c::ComplexF64, b::BasisOperator)::ComplexF64
    return a.terms[b] = c
end

function iterate(a::Operator)
    iterate(a.terms)
end

function iterate(a::Operator, i)
    iterate(a.terms, i)
end

function in(b::BasisOperator, a::Operator)::Bool
    return b in keys(a.terms)
end

function zero(::Type{Operator})::Operator
    return Operator()
end

function zero(::Operator)::Operator
    zero(Operator)
end

function isfermion(p::PauliMode)
    return false
end

function isfermion(d::DiracMode)
    return d.cr ⊻ d.an
end

function isfermion(m::MajoranaMode)
    return m.n
end

function isfermion(b::BoseMode)
    return false
end

function isfermion(b::BasisOperator)
    f = false
    for d in a.diracs
        f ⊻= isfermion(d)
    end
    for m in a.majoranas
        f ⊻= isfermion(m)
    end
    return f
end

function mmul(cb, a::PauliMode, b::PauliMode)
    if a.pauli == σI
        cb(PauliMode(b.pauli), 1.0)
    elseif b.pauli == σI
        cb(PauliMode(a.pauli), 1.0)
    elseif a.pauli == b.pauli
        cb(PauliMode(σI), 1.0)
    elseif a.pauli == σX && b.pauli == σY
        cb(PauliMode(σZ), 1.0im)
    elseif a.pauli == σY && b.pauli == σZ
        cb(PauliMode(σX), 1.0im)
    elseif a.pauli == σZ && b.pauli == σX
        cb(PauliMode(σY), 1.0im)
    elseif b.pauli == σX && a.pauli == σY
        cb(PauliMode(σZ), -1.0im)
    elseif b.pauli == σY && a.pauli == σZ
        cb(PauliMode(σX), -1.0im)
    elseif b.pauli == σZ && a.pauli == σX
        cb(PauliMode(σY), -1.0im)
    end
end

function mmul(cb, a::DiracMode, b::DiracMode)
    if a.an == false
        if a.cr && b.cr
            return
        end
        cb(DiracMode(a.cr | b.cr, b.an), 1.0)
        return
    end
    if b.cr == false
        if a.an && b.an
            return
        end
        cb(DiracMode(a.cr, a.an | b.an), 1.0)
        return
    end
    # a adag = 1 - adag a
    cb(DiracMode(a.cr, b.an), 1.0)
    if !a.cr && !b.an
        cb(DiracMode(true, true), -1.0)
    end
end

function mmul(cb, a::MajoranaMode, b::MajoranaMode)
    if a.n && b.n
        return
    end
    cb(MajoranaMode(a.n | b.n), 1.0)
end

function mmul(cb, a::BoseMode, b::BoseMode)
    if a.an == 0
        cb(BoseMode(a.cr+b.cr,b.an), 1.0)
        return
    end
    if b.cr == 0
        cb(BoseMode(a.cr,a.an+b.an), 1.0)
        return
    end
    # Commute one of b.cr through.
    # (c^k) c⁺ = c⁺ c^k + k c^{k-1}
    k = a.an
    mmul(cb, BoseMode(a.cr+1, a.an), BoseMode(b.cr-1, b.an))
    mmul(BoseMode(a.cr, a.an-1), BoseMode(b.cr-1, b.an)) do m, coef
        cb(m, coef*k)
    end
end

function bmul(cb, a::BasisOperator, b::BasisOperator)
    # First we have to swap everything into place.
    nfa::Int = sum(map(isfermion, a.diracs)) + sum(map(isfermion, a.majoranas))
    nswaps::Int = 0
    for (ma,mb) in zip(a.diracs,b.diracs)
        nfa -= isfermion(ma)
        if isfermion(mb)
            nswaps += nfa
        end
    end
    for (ma,mb) in zip(a.majoranas,b.majoranas)
        nfa -= isfermion(ma)
        if isfermion(mb)
            nswaps += nfa
        end
    end
    # Now record all factors.
    paulis = Vector{Vector{Tuple{PauliMode,ComplexF64}}}(undef, length(a.paulis))
    diracs = Vector{Vector{Tuple{DiracMode,ComplexF64}}}(undef, length(a.diracs))
    majoranas = Vector{Vector{Tuple{MajoranaMode,ComplexF64}}}(undef, length(a.majoranas))
    boses = Vector{Vector{Tuple{BoseMode,ComplexF64}}}(undef, length(a.boses))
    for i in 1:length(a.paulis)
        paulis[i] = Tuple{PauliMode,ComplexF64}[]
        mmul(a.paulis[i], b.paulis[i]) do m,c
            push!(paulis[i], (m,c))
        end
    end
    for i in 1:length(a.diracs)
        diracs[i] = Tuple{DiracMode,ComplexF64}[]
        mmul(a.diracs[i], b.diracs[i]) do m,c
            push!(diracs[i], (m,c))
        end
    end
    for i in 1:length(a.majoranas)
        majoranas[i] = Tuple{MajoranaMode,ComplexF64}[]
        mmul(a.majoranas[i], b.majoranas[i]) do m,c
            push!(majoranas[i], (m,c))
        end
    end
    for i in 1:length(a.boses)
        boses[i] = Tuple{BoseMode,ComplexF64}[]
        mmul(a.boses[i], b.boses[i]) do m,c
            push!(boses[i], (m,c))
        end
    end
    for x in Iterators.product(paulis..., diracs..., majoranas..., boses...)
        k = 0
        r = copy(a)
        c::ComplexF64 = (-1.0)^nswaps
        for i in 1:length(a.paulis)
            k += 1
            r.paulis[i] = x[k][1]
            c *= x[k][2]
        end
        for i in 1:length(a.diracs)
            k += 1
            r.diracs[i] = x[k][1]
            c *= x[k][2]
        end
        for i in 1:length(a.majoranas)
            k += 1
            r.majoranas[i] = x[k][1]
            c *= x[k][2]
        end
        for i in 1:length(a.boses)
            k += 1
            r.boses[i] = x[k][1]
            c *= x[k][2]
        end
        cb(r, c)
    end
end

function badjoint(cb, a::BasisOperator)
    # Number of fermionic modes.
    nf::Int = 0
    b = copy(a)
    # No change needs to be made to Pauli or Majorana. Dirac and Bose modes are
    # not self-adjoint.
    for m in a.majoranas
        nf += isfermion(m)
    end
    for (i,m) in enumerate(a.diracs)
        nf += isfermion(m)
        b.diracs[i] = DiracMode(m.an,m.cr)
    end
    for (i,m) in enumerate(a.boses)
        b.boses[i] = BoseMode(m.an,m.cr)
    end
    # If there are n fermionic modes, the number of swaps to perform is (n-1) +
    # (n-2) + ... + 1. All that matters is whether this is even or odd.
    c::ComplexF64 = if nf%4 == 2 || nf%4 == 3
        -1
    else
        1
    end
    cb(b,c)
end

macro algebra(name, block)
    @assert block.head == :block
    gens = []
    idn = nothing
    for line in block.args
        if typeof(line) == LineNumberNode
            continue
        end
        if line.head != :(::)
            @error "Unexpected something"
        end
        n = line.args[1]
        if typeof(line.args[2]) == Expr
            if line.args[2].head != :ref
                @error "Unexpected something"
            end
            t, K = line.args[2].args
            push!(gens, (n, t, (K,)))
        else
            t = line.args[2]
            if t != :Identity
                push!(gens, (n, t, ()))
            else
                idn = n
            end
        end
    end
    def = quote
    end
    function append(e)
        def = quote
            $def
            $e
        end
    end
    def = quote
        n = Dict{Symbol,Int}()
        n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose] = 0, 0, 0, 0
    end
    for (n, t, s) in gens
        k = if s == ()
            1
        else
            s[1]
        end
        append(:(n[$(esc(QuoteNode(t)))] += $(esc(k))))
    end
    def = quote
        $def
        proto = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
        i = Dict{Symbol,Int}()
        i[:Pauli], i[:Dirac], i[:Majorana], i[:Bose] = 0, 0, 0, 0
    end
    function build(t)
        if t == :Pauli
            quote
                bx = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                by = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                bz = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                bx.paulis[i[$(QuoteNode(t))]] = PauliMode(σX)
                by.paulis[i[$(QuoteNode(t))]] = PauliMode(σY)
                bz.paulis[i[$(QuoteNode(t))]] = PauliMode(σZ)
                [Operator(bx),Operator(by),Operator(bz)]
            end
        elseif t == :Dirac
            quote
                base = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                base.diracs[i[$(QuoteNode(t))]] = DiracMode(false,true)
                Operator(base)
            end
        elseif t == :Majorana
            quote
                base = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                base.majoranas[i[$(QuoteNode(t))]] = MajoranaMode(true)
                Operator(base)
            end
        elseif t == :Bose
            quote
                base = BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose])
                base.boses[i[$(QuoteNode(t))]] = BoseMode(0,1)
                Operator(base)
            end
        end
    end
    for (n, t, s) in gens
        if s == ()
            append(:(i[$(QuoteNode(t))] += 1))
            append(:($(esc(n)) = $(build(t))))
        else
            @assert length(s) == 1
            K = s[1]
            initexpr = :(Vector{Operator}(undef, K))
            if t == :Pauli
                initexpr = :(Vector{Vector{Operator}}(undef, K))
            end
            def = quote
                $def
                K = $(esc(K))
                $(esc(n)) = $(initexpr)
                for k in 1:K
                    i[$(QuoteNode(t))] += 1
                    $(esc(n))[k] = $(build(t))
                end
            end
        end
    end
    if idn != nothing
        def = quote
            $def
            $(esc(idn)) = Operator(BasisOperator(n[:Pauli], n[:Dirac], n[:Majorana], n[:Bose]))
        end
    end
    def
end

macro check(expr)
    buf = IOBuffer()
    Base.show_unquoted(buf, expr)
    str = String(take!(buf))
    quote
        r::Bool = $(esc(expr))
        if !r
            printstyled("    Test failed: $($(str))\n", color=:red)
        end
        r
    end
end

function selftest()
    printstyled("Beginning self-test\n", bold=true)
    Random.seed!(0)

    printstyled("  Isolated tests: Pauli\n", bold=true)
    @algebra SinglePauli begin
        σ::Pauli
    end
    @check σ[1] ≈ σ[1]
    @check !(σ[1] ≈ σ[2])
    @check adjoint(σ[1]) ≈ σ[1]
    @check σ[1] * σ[1] ≈ σ[1] * σ[1]
    @check σ[1] * σ[1] ≈ σ[2] * σ[2]
    @check σ[1] * σ[2] ≈ 1im * σ[3]
    @check !(σ[1] * σ[2] ≈ -1im * σ[3])
    @algebra PauliAlgebra begin
        σ::Pauli[3]
    end
    for (i,j) in zip(1:3,1:3)
        @check σ[1][i] * σ[2][j] ≈ σ[2][j] * σ[1][i]
    end

    printstyled("  Isolated tests: Dirac\n", bold=true)
    @algebra SingleDirac begin
        a::Dirac
    end
    @check a ≈ a
    @check !(adjoint(a) ≈ a)
    @check adjoint(a) ≈ adjoint(a)
    @algebra TwoDiracs begin
        I::Identity
        a::Dirac
        b::Dirac
    end
    @check a * b ≈ -b * a
    @check !(a * b ≈ b * a)
    @check a * a ≈ 0*a
    @check !(adjoint(a) * a ≈ a * adjoint(a))
    @check I - adjoint(a) * a ≈ a * adjoint(a)
    @algebra DiracAlgebra begin
        a::Dirac[8]
    end
    @check a[1] * a[2] ≈ - a[2] * a[1]
    @check a[1] * a[2] * adjoint(a[3]) ≈ - adjoint(a[3]) * a[2] * a[1]
    @check a[1] * a[2] * adjoint(a[3]) * a[4] ≈ a[4] * adjoint(a[3]) * a[2] * a[1]
    @check adjoint(a[1])*a[1] * a[2] ≈ a[2] * adjoint(a[1]) * a[1]

    printstyled("  Isolated tests: Majorana\n", bold=true)
    @algebra SingleMajorana begin
        γ::Majorana
    end
    @check !(γ ≈ γ*γ)
    @check γ*γ ≈ γ*γ*γ
    @algebra MajoranaAlgebra begin
        γ::Majorana[8]
    end
    for i in 1:8
        for j in 1:8
            @check γ[i] * γ[j] ≈ -γ[j] * γ[i]
        end
    end
    for i in 1:8
        for j in 1:8
            for k in 1:8
                @check γ[i] * (γ[j] * γ[k]) ≈ (γ[i] * γ[j]) * γ[k]
            end
        end
    end

    printstyled("  Isolated tests: Bose\n", bold=true)
    @algebra SingleBose begin
        I::Identity
        c::Bose
    end
    @check I*c ≈ c*I
    @check c*c ≈ c*c
    @check adjoint(c*c) ≈ adjoint(c) * adjoint(c)
    @check c*adjoint(c) ≈ I + adjoint(c) * c
    @check c*c*adjoint(c) ≈ c + c*adjoint(c)*c
    @check c*adjoint(c)*adjoint(c) ≈ adjoint(c*c*adjoint(c))

    @algebra BoseAlgebra begin
        I::Identity
        c::Bose[8]
    end

    K::Int = 6
    @algebra BigAlgebra begin
        σ::Pauli[K]
        a::Dirac[K]
        γ::Majorana[K]
        c::Bose[K]
    end
    function random()::Operator
        op = σ[1][1]
        for k in 1:K
            i = rand(0:3)
        end
        return op
    end
    printstyled("  Randomized tests: associativity\n", bold=true)
end

function main()
    selftest()
end

main()

