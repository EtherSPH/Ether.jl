#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:30:26
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * Val(x) is allowed as x can be inferred at compile time.
 =#

include("../../head/oneapi.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline f(::Val{1}) = 2
@inline f(::Val{2}) = 1

@kernel function ker(::Type{Dimension}, x) where {N, Dimension <: AbstractDimension{N}}
    I = @index(Global)
    x[I] = f(Val(N))
    # x[I] = f(Val(I)) # this line does not work
end

a = [1, 2] |> CT
ker(Backend, 2)(Dimension2D, a, ndrange = (2,))
KernelAbstractions.synchronize(Backend)
@info a
