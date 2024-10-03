module TestSplines

using Test

using Random: rand!, randn!

using CONCAVE
using CONCAVE.Splines

@testset "small quadratic splines" begin
    spline0 = QuadraticSpline(1.0, 0)
    spline1 = QuadraticSpline(1.0, 1)
    spline2 = QuadraticSpline(1.0, 2)


    spline0.c[1] = 0
    spline0.c[2] = 1
    spline0.c[3] = 0
    at!(spline0, 0.0)
    @test spline0.f == 0
    @test spline0.∂c[1] == 1
    @test spline0.∂c′[1] == 0
    @test spline0.∂c′[2] == 1
    at!(spline0, 0.5)
    @test spline0.f == 0.5
    @test spline0.∂c[1] == 1

    at!(spline1, 0.0)
    @test spline1.∂c[1] == 1
    at!(spline1, 1.0)
    @test spline1.∂c[1] == 1

    at!(spline2, 1.0)
    @test spline2.∂c[1] == 1
end

@testset "random quadratic splines: integral" begin
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

@testset "random quadratic spline: derivative of integral" begin
    for n in 1:1000
        K = rand(1:20)
        T = rand()*50
        x = rand()*T
        spline = QuadraticSpline(T, K)
        rand!(spline.c)
        t = rand()*T
        ϵ = 1e-4

        at!(spline, t)
        j = rand(1:length(spline.∂∫))
        ∂∫ = spline.∂∫[j]
        spline.c[j] += ϵ
        at!(spline, t)
        ∫₊ = spline.∫
        spline.c[j] -= 2*ϵ
        at!(spline, t)
        ∫₋ = spline.∫
        @test abs((∫₊ - ∫₋)/(2*ϵ) - ∂∫)/(1e-2+abs(∂∫)) < 1e-4
    end
end

end
