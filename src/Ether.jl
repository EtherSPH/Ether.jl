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
include("VTK/VTK.jl")
include("SPH/SPH.jl")

export Utility
export Environment, Math
export Geometry
export Macro
export Class, Algorithm
export DataIO, VTK
export SPH

using FIGlet

function __init__()
    println("="^100)
    println("`Ether.jl` is a particle-based simulation framework running on both cpu and gpu supports:")
    for (backend, color) in Environment.kNameToColor
        printstyled("$backend  ", color = color)
    end
    println()
    FIGlet.render("Ether", "ANSI Shadow")
    println("="^100)
end

end # module Ether
