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

    o = 0
    function coef()
        o += 1
        return s.c[o]
    end

    s.f = coef()
    s.f′ = coef()
    f′′ = coef()
    t′ = 0
    function advance(dt::Float64)
        s.∫ += f′′ * dt^3 / 6 + s.f′ * dt^2 / 2 + s.f * dt
        # TODO ∂∫
        s.f += dt * s.f′ + f′′ * dt^2 / 2
        # TODO ∂c
        s.f′ += dt * f′′
        # TODO ∂c′
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
