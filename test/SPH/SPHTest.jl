#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:43:25
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "SPH" begin
    include("KernelTest.jl")
    include("LibraryTest.jl")
end
