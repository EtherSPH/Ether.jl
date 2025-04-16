#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 20:31:20
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Optimized" begin
    for i in 1:10
        @test Math.power(2, Val(i)) == 2^i
    end
end
