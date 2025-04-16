#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 18:19:22
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Math" begin
    include("MeanTest.jl")
    include("OptimizedTest.jl")
end
