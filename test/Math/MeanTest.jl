#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 18:21:56
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Mean" begin
    a = 1.0f0
    b = 4.0f0
    @test Math.Mean.arithmetic(a, b) ≈ 2.5f0
    @test Math.Mean.invarithmetic(a, b) ≈ 1 / 2.5f0
    @test Math.Mean.geometric(a, b) ≈ 2.0f0
    @test Math.Mean.invgeometric(a, b) ≈ 1 / 2.0f0
    @test Math.Mean.harmonic(a, b) ≈ 1.6f0
    @test Math.Mean.invharmonic(a, b) ≈ 1 / 1.6f0
    @test Math.Mean.quadratic(a, b) ≈ sqrt(17.0f0)
    @test Math.Mean.invquadratic(a, b) ≈ 1 / sqrt(17.0f0)
end
