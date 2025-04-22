#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:25:28
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * NamedTuple on GPU is allowed.
            # ! on Intel UHD Graphics 620 @ 1.15 GHz [Integrated]
 =#

include("../../head/oneapi.jl")

@kernel function device_vadd!(a, x::NamedTuple)
    I = @index(Global)
    a[I] += x.a
    a[I] *= x.b
end

function host_vadd!(a, x::NamedTuple)
    device_vadd!(Backend, 2)(a, x, ndrange = (2,))
    KernelAbstractions.synchronize(Backend)
end

a = randn(FT, 2) |> CT
x = (a = 1.0f0, b = IT(2), c = 3.0f0)

@info "before:"
@info "a = $(a)"
host_vadd!(a, x)
@info "after vadd:"
@info "a = $(a)"
println(x)
