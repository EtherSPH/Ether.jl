#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 15:25:01
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using KernelAbstractions
using Random

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
