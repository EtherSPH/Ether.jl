#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:26:39
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Class

using KernelAbstractions

using Ether.Environment

const kDefaultThreadNumber = 256
const kDefaultMaxNeighbourNumber = 40

include("Domain/Domain.jl")
include("ParticleSystem/ParticleSystem.jl")
include("NeighbourSystem/NeighbourSystem.jl")

export AbstractDomain
export AbstractParticleSystem, AbstractHostParticleSystem
export AbstractNeighbourSystem
export AbstractPeriodicBoundary, NonePeriodicBoundary, PeriodicBoundary2D, PeriodicBoundary3D

end # module Class
