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
    i = 0
    function coef()
        i += 1
        return s.c[i]
    end
    f = coef()
    f′ = coef()
    f′′ = coef()
    t′ = 0
    for k in 1:s.K
        if t < c.x[k]
            return
        end
        dt = c.x[k] - t′
        t′ = c.x[k]
        f = dt*f′
        f′ = dt*f′′
        f′′ = coef()
    end
    # TODO
end

end
