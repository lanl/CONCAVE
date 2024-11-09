module TestUnconstrained

using LinearAlgebra
using Test

using CONCAVE
using CONCAVE.UnconstrainedOptimization: Newton,minimize!
using CONCAVE.Utilities: check_gradients

@testset verbose=true "Newton" begin
    @testset "Quadratic" begin
        for k in 1:100
            N = rand(1:10)
            b = randn(Float64, N)
            M = randn(Float64, (N,N))
            M = 0.5 * (M + M')
            M += (0.01 - minimum(eigvals(M)))*I

            function f!(g, h, y)::Float64
                if !isnothing(h)
                    @. h = 2*M
                end
                if !isnothing(g)
                    g .= 2 .* M * y .+ b
                end
                return (y' * M * y) + b' * y
            end

            y = randn(Float64, N)
            let
                g = zeros(Float64, N)
                h = zeros(Float64, (N,N))
                f!(g, h, y)
                @test check_gradients(y, g, h) do x
                    f!(nothing, nothing, x)
                end
            end

            opt = minimize!(f!, Newton, y)
            opt′ = -1/4 * b' * inv(M) * b
            @test abs(opt - opt′) / (abs(opt + opt′)/2) < 1e-6
        end
    end

    @testset "Quartic" begin
        for k in 1:100
            N = rand(1:5)
            b = randn(Float64, N)
            M₂ = randn(Float64, (N,N))
            M₂ = 0.5 * (M₂ + M₂')
            M₂ += (0.01 - minimum(eigvals(M₂)))*I
            M₄ = randn(Float64, (N,N))
            M₄ = 0.5 * (M₄ + M₄')
            M₄ += (0.01 - minimum(eigvals(M₄)))*I

            function f!(g, h, y)::Float64
                if !isnothing(g)
                    g .= 0.
                    @. g += b
                    g .+= 2 .* M₂ * y
                    g .+= 2 .* (y' * M₄ * y) * 2 * M₄ * y
                end
                if !isnothing(h)
                    h .= 0.
                    @. h += 2 * M₂
                    for i in 1:N, j in 1:N
                        h[i,j] += 2 * (y' * M₄ * y) * 2 * M₄[i,j]
                        h[i,j] += 2 * 2 * 2 * (M₄ * y)[i] * (M₄ * y)[j]
                    end
                end
                return (y' * M₄ * y)^2 + (y' * M₂ * y) + b' * y
            end

            y = randn(Float64, N)
            let
                g = zeros(Float64, N)
                h = zeros(Float64, (N,N))
                f!(g, h, y)
                @test check_gradients(y, g, h) do x
                    f!(nothing, nothing, x)
                end
            end

            opt = minimize!(f!, Newton, y)
        end
    end
end

end
