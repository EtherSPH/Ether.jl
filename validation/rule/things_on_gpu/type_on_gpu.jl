#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:29:26
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * Type parsing on GPU is allowed.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
 =#

include("../../head/oneapi.jl")

abstract type AbstractDimension{N} end

struct Dimension1D <: AbstractDimension{1} end
struct Dimension2D <: AbstractDimension{2} end
struct Dimension3D <: AbstractDimension{3} end

@inline function dimension(::Type{<:AbstractDimension{N}}) where {N}
    return N
end

@inline function vadd!(::Type{Dimension2D}, I, a, b)
    i = 1
    while i <= 2
        a[I, i] += b[I, i]
        i += 1
    end
end

@inline function vadd!(::Type{Dimension3D}, I, a, b)
    i = 1
    while i <= 3
        a[I, i] += b[I, i]
        i += 1
    end
end

@kernel function device_vadd!(Dimension, a, b)
    I = @index(Global)
    vadd!(Dimension, I, a, b)
end

a = CT(randn(FT, 2, 3))
b = CT(randn(FT, 2, 3))

@info "before:"
@info "a = $(a)"
@info "b = $(b)"
device_vadd!(Backend, 2)(Dimension2D, a, b, ndrange = (2,))
KernelAbstractions.synchronize(Backend)
@info "after vadd Dimension2D:"
@info "a = $(a)"
device_vadd!(Backend, 2)(Dimension3D, a, b, ndrange = (2,))
KernelAbstractions.synchronize(Backend)
@info "after vadd Dimension3D:"
@info "a = $(a)"
