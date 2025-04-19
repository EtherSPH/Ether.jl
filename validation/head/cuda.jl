#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 15:25:14
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using KernelAbstractions
using CUDA
using Random

const IT = Int32
const FT = Float32
const CT = CUDA.CuArray
const Backend = CUDA.CUDABackend()
