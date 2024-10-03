module TestSplines

using Test

using Random: rand!, randn!

using CONCAVE
using CONCAVE.Splines

@testset "small quadratic splines" begin
    spline0 = QuadraticSpline(1.0, 0)
    spline1 = QuadraticSpline(1.0, 1)
    spline2 = QuadraticSpline(1.0, 2)

    at!(spline0, 0.0)
end

@testset "random quadratic splines" begin
    for n in 1:100
        K = rand(1:20)
        T = rand()*50
        x = rand()*T
        spline = QuadraticSpline(T, K)
        rand!(spline.c)

        t = rand()*T

        ϵ = 1e-4
        at!(spline, t)
        f = spline.f
        at!(spline, t-ϵ)
        ∫₋ = spline.∫
        at!(spline, t+ϵ)
        ∫₊ = spline.∫
        @test abs((∫₊ - ∫₋)/(2*ϵ) - f)/abs(f) < 1e-4
    end
end

end
