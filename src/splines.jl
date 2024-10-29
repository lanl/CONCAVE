module Splines

export QuadraticSpline
export at!

mutable struct QuadraticSpline
    K::Int

    # Parameters defining the spline.
    c::Vector{Float64}
    x::Vector{Float64}

    # Value and first derivative
    f::Float64
    f′::Float64

    # Derivatives of value with respect to parameters
    ∂c::Vector{Float64}

    # Derivatives of derivative with respect to parameters
    ∂c′::Vector{Float64}

    # Integral
    ∫::Float64

    # Derivative of the integral with respect to parameters
    ∂∫::Vector{Float64}

    function QuadraticSpline(T::Float64, K::Int)
        c = zeros(Float64, K+3)
        ∂c = zero(c)
        ∂c′ = zero(c)
        ∂∫ = zero(c)

        x = zeros(Float64, K)
        for k in 1:K
            x[k] = k*T/(K+1)
        end

        new(K, c, x, 0., 0., ∂c, ∂c′, 0., ∂∫)
    end
end

function at!(s::QuadraticSpline, t::Float64)
    s.∫ = 0.
    s.∂c .= 0.
    s.∂c′ .= 0.
    s.∂∫ .= 0.

    o::Int = 0
    function coef()::Float64
        o += 1
        return s.c[o]
    end

    s.f = coef()
    s.∂c[o] = 1.0
    s.f′ = coef()
    s.∂c′[o] = 1.0
    f′′::Float64 = coef()
    t′::Float64 = 0
    function advance(dt::Float64)
        s.∫ += f′′ * dt^3 / 6 + s.f′ * dt^2 / 2 + s.f * dt
        s.∂∫ .+= dt .* s.∂c
        s.∂∫ .+= (dt^2 / 2) .* s.∂c′
        s.∂∫[o] += dt^3 / 6

        s.f += dt * s.f′ + f′′ * (dt^2 / 2)
        s.∂c .+= dt .* s.∂c′
        s.∂c[o] += dt^2 / 2

        s.f′ += dt * f′′
        s.∂c′[o] += dt

        t′ += dt
    end
    for k in 1:s.K
        if t < s.x[k]
            advance(t - t′)
            return
        end
        advance(s.x[k] - t′)
        f′′ = coef()
    end
    advance(t - t′)
end

end
