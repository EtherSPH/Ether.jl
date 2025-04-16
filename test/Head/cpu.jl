#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:03:24
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Ether
using KernelAbstractions

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
const parallel = Ether.Environment.Parallel{IT, FT, CT, Backend}()
const parallel_cpu = Ether.Environment.ParallelCPU{IT, FT}()
