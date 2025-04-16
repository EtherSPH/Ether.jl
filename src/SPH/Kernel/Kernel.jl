#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:37:53
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Kernel

# * see reference [pysph](https://pysph.readthedocs.io/en/latest/reference/kernels.html)

using Ether.Math

abstract type AbstractKernel{IT <: Integer, FT <: AbstractFloat, N} end
export AbstractKernel

include("CubicSpline.jl")
include("Gaussian.jl")
include("WendlandC2.jl")
include("WendlandC4.jl")

@inline @fastmath function _value0(
    hinv::Real,
    kernel::AbstractKernel{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return sigma(kernel) * Math.power(hinv, Val(N))
end

@inline @fastmath function value0(
    h::Real,
    kernel::AbstractKernel{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value0(1 / h, kernel)
end

const W = value
const DW = gradient
const _W = _value
const _DW = _gradient
const _W0 = _value0

end # module Kernel
