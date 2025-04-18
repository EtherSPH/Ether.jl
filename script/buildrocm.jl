#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/18 16:37:54
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Pkg
Pkg.add("AMDGPU")
Pkg.build()
using AMDGPU
AMDGPU.versioninfo()
