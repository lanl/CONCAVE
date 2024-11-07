module Utilities

function check_gradients(f, y, g, h)::Bool
    ϵ = 1e-5
    N = length(y)
    @assert size(g) == (N,)
    for n in 1:N
        y₀ = y[n]
        y[n] = y₀ - ϵ
        x₋ = f(y)
        y[n] = y₀ + ϵ
        x₊ = f(y)
        y[n] = y₀
        gn = (x₊ - x₋) / (2 * ϵ)
        if abs(gn - g[n]) / abs(gn + g[n] + 1e-10) > ϵ
            return false
        end
    end
    if isnothing(h)
        return true
    end
    @assert size(h) == (N,N)
    bad = false
    for i in 1:N, j in 1:N
        yᵢ₀ = y[i]
        yⱼ₀ = y[j]
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] += ϵ
        y[j] += ϵ
        x₊₊ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] += ϵ
        y[j] -= ϵ
        x₊₋ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] -= ϵ
        y[j] += ϵ
        x₋₊ = f(y)
        y[i], y[j] = yᵢ₀, yⱼ₀
        y[i] -= ϵ
        y[j] -= ϵ
        x₋₋ = f(y)
        g₊ = (x₊₊ - x₊₋) / (2*ϵ)
        g₋ = (x₋₊ - x₋₋) / (2*ϵ)
        hij = (g₊ - g₋)/(2*ϵ)
        if abs(hij - h[i,j]) / (maximum(abs.(h)) + 1e-8) > sqrt(ϵ)
            return false
        end
    end
    return true
end

end
