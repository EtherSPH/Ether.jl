#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 16:19:55
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Algorithm" begin
    include("ActionTest.jl")
    include("SearchTest.jl")
end
