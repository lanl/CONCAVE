module Utilities

using Printf

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

# Output matrices for processing in mathematica
function print_mathematica(x::Float64)
    expon = 0
    while abs(x) > 10
        x /= 10
        expon += 1
    end
    while abs(x) < 1 && abs(expon) < 10
        expon -= 1
        x *= 10
    end
    @printf "%f*10^(%d)" x expon
end
function print_mathematica(x::ComplexF64)
    print_mathematica(real(x))
    if imag(x) < 0
        print("-")
        print_mathematica(-imag(x))
    else
        print("+")
        print_mathematica(imag(x))
    end
    print("I")
end
function print_mathematica(mat::Matrix)
    N = size(mat)[1]
    print("{")
    for i in 1:N
        print("{")
        for j in 1:N
            print_mathematica(mat[i,j])
            if j < N
                print(" , ")
            end
        end
        print("}")
        if i < N
            print(" , ")
        end
    end
    print("}")
end

end
