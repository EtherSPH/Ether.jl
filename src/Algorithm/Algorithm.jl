#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 15:03:49
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Algorithm

using Atomix
using KernelAbstractions

using Ether.Environment
using Ether.Math
using Ether.Class

abstract type AbstractLaunch end
struct Static <: AbstractLaunch end
struct Dynamic <: AbstractLaunch end

const kDefaultThreadNumber = 256
const kDefaultParameter = (t = 0,)
const kDefaultLaunch = Static()
const kStatic = Static()
const kDynamic = Dynamic()

include("Action.jl")
include("Periodic.jl")
include("Search.jl")

end # module Algorithm
