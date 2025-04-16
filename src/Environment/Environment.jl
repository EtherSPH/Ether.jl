#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 14:13:52
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Environment

using KernelAbstractions

include("Dimension.jl")
include("Backend.jl")
include("Parallel.jl")

export AbstractDimension
export Dimension1D, Dimension2D, Dimension3D
export AbstracTensor
export Tscalar, Tvector, Tmatrix
export capacity

export AbstractParallel

end # module Environment
