#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:46:14
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Class" begin
    include("DomainTest.jl")
    include("ParticleSystem/ParticleSystemTest.jl")
    include("NeighbourSystem/NeighbourSystemTest.jl")
end
