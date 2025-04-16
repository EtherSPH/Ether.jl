#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:06:23
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Environment" begin
    include("DimensionTest.jl")
    include("ParallelTest.jl")
end
