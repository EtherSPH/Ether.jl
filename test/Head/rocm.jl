#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:05:12
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Ether
using KernelAbstractions
import Pkg
Pkg.add("AMDGPU")
using AMDGPU

const IT = Int32
const FT = Float32
const CT = AMDGPU.ROCArray
const Backend = AMDGPU.ROCBackend()
const parallel = Ether.Environment.Parallel{IT, FT, CT, Backend}()
const parallel_cpu = Ether.Environment.ParallelCPU{IT, FT}()
