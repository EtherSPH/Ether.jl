#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:33:38
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== AbstractDomain2D ==================== *

abstract type AbstractDomain3D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension3D} end

# * ==================== Domain3D ==================== *

# TODO: add 3D support
