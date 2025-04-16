#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:06:37
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Dimension" begin
    @test Environment.dimension(Environment.Dimension1D) == 1
    @test Environment.dimension(Environment.Dimension2D) == 2
    @test Environment.dimension(Environment.Dimension3D) == 3
    @test Environment.order(Environment.Tscalar) == 0
    @test Environment.order(Environment.Tvector) == 1
    @test Environment.order(Environment.Tmatrix) == 2
    @test Environment.capacity(Val(Environment.Dimension1D), Val(Environment.Tscalar)) == 1
    @test Environment.capacity(Val(Environment.Dimension2D), Val(Environment.Tscalar)) == 1
    @test Environment.capacity(Val(Environment.Dimension3D), Val(Environment.Tscalar)) == 1
    @test Environment.capacity(Val(Environment.Dimension1D), Val(Environment.Tvector)) == 1
    @test Environment.capacity(Val(Environment.Dimension2D), Val(Environment.Tvector)) == 2
    @test Environment.capacity(Val(Environment.Dimension3D), Val(Environment.Tvector)) == 3
    @test Environment.capacity(Val(Environment.Dimension1D), Val(Environment.Tmatrix)) == 1
    @test Environment.capacity(Val(Environment.Dimension2D), Val(Environment.Tmatrix)) == 4
    @test Environment.capacity(Val(Environment.Dimension3D), Val(Environment.Tmatrix)) == 9
end
