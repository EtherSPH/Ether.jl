#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:24:52
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # ! global vars on GPU is not allowed.
 =#

include("../../head/oneapi.jl")

t = 1.0f0

@kernel function device_vadd!(a)
    global t
    I = @index(Global)
    a[I] += t
end

function host_vadd!(a)
    device_vadd!(Backend, 2)(a, ndrange = (2,))
    KernelAbstractions.synchronize(Backend)
end

a = zeros(FT, 2) |> CT

@info "before:"
@info "a = $(a)"
host_vadd!(a)
@info "after vadd:"
@info "a = $(a)"
