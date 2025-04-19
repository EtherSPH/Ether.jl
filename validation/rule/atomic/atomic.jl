#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 15:23:53
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # ! info: `oneAPI.jl` currently only supports atomic operations for `Int32` types.
 =#

include("../../head/oneapi.jl")

using Atomix

const n = 10

a = Vector{IT}(1:n) |> CT
b = zeros(IT, 1) |> CT
c = zeros(IT, n) |> CT

@kernel function device_atomic_add!(a, b, c)
    I = @index(Global)
    c[I] = Atomix.@atomic b[1] += a[I]
end

function host_atomic_add!(a, b, c)
    device_atomic_add!(Backend, n)(a, b, c, ndrange = (n,))
    KernelAbstractions.synchronize(Backend)
end

host_atomic_add!(a, b, c)
@info "b = $(b)"
@info "c = $(c)"
