#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:19:04
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Geometry

abstract type AbstractGeometry{N} end

@inline dimension(::AbstractGeometry{N}) where {N} = N

include("Geometry2D.jl")
include("Geometry3D.jl")

end # module Geometry
