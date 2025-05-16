#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/11 21:04:35
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Geometry 3D" begin
    # * cuboid
    cuboid = Geometry.Cuboid(0.0, 0.0, 0.0, 2.0, 3.0, 4.0)
    @test Geometry.count(0.1, cuboid) == 20 * 30 * 40
    @test Geometry.inside(0.1, 0.2, 0.3, cuboid) == true
    @test Geometry.inside(2.1, 0.2, 0.3, cuboid) == false
end
