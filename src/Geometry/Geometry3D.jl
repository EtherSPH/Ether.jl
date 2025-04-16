#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:19:46
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

struct Cuboid{T <: Real} <: AbstractGeometry{3}
    first_x_::T
    first_y_::T
    first_z_::T
    last_x_::T
    last_y_::T
    last_z_::T
    span_x_::T
    span_y_::T
    span_z_::T
end
