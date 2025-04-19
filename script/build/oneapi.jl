#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/18 16:39:12
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Pkg
Pkg.add("oneAPI")
Pkg.build()
using oneAPI
oneAPI.versioninfo()
