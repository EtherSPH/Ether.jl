#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/18 16:39:31
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Pkg
Pkg.add("Metal")
Pkg.build()
using Metal
Metal.macos_version()
