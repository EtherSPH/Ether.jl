#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 14:18:51
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline function dimension(::Type{Dimension}) where {N, Dimension <: AbstractDimension{N}}
    return N
end

abstract type AbstractTensor{M} end
struct Tscalar <: AbstractTensor{0} end
struct Tvector <: AbstractTensor{1} end
struct Tmatrix <: AbstractTensor{2} end

@inline function order(::Type{Tensor}) where {M, Tensor <: AbstractTensor{M}}
    return M
end

@inline function capacity(
    ::Val{Dimension},
    ::Val{Tensor},
) where {N, M, Dimension <: AbstractDimension{N}, Tensor <: AbstractTensor{M}}
    return N^M
end
