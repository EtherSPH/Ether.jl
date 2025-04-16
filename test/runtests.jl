#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:02:10
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Test
using KernelAbstractions
using OrderedCollections
using Ether
using Ether.Macro

# support for `cpu`, `cuda`, `rocm`, `oneapi`, `metal`
const DEVICE = "oneapi"
include("Head/$(DEVICE).jl")
@info "test on backend: $DEVICE"

@testset "EtherParallelParticles" begin
    include("Utility/UtilityTest.jl")
    include("Environment/EnvironmentTest.jl")
    include("Math/MathTest.jl")
    include("Geometry/GeometryTest.jl")
    include("Class/ClassTest.jl")
    include("Macro/MacroTest.jl")
    include("Algorithm/AlgorithmTest.jl")
    include("DataIO/DataIOTest.jl")
    include("SPH/SPHTest.jl")
end
