#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:28:14
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * ops on gpu is allowed
 =#

include("../../head/oneapi.jl")

@inline function apply_op(op::Function, x, y)
    return op(x, y)
end

@kernel function some_op(x)
    I = @index(Global)
    @inbounds x[I, 1] = apply_op(+, x[I, 2], x[I, 3])
end

a = rand(Float32, 4, 3) |> CT
@info a
some_op(Backend, 4)(a, ndrange = (4,))
KernelAbstractions.synchronize(Backend)
@info a
