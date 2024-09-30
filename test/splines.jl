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
        k = rand(1:20)
        T = rand()*50
        x = rand()*T
        spline = QuadraticSpline(T, k)
    end
end

end
