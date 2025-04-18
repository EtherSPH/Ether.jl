#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 14:10:54
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Ether

include("Utility/Utility.jl")
include("Environment/Environment.jl")
include("Geometry/Geometry.jl")
include("Math/Math.jl")
include("Macro/Macro.jl")
include("Class/Class.jl")
include("Algorithm/Algorithm.jl")
include("DataIO/DataIO.jl")
include("SPH/SPH.jl")

export Utility
export Environment, Math
export Geometry
export Macro
export Class, Algorithm
export DataIO
export SPH

using FIGlet

function __init__()
    FIGlet.render("Ether.jl", "standard")
end

end # module Ether
