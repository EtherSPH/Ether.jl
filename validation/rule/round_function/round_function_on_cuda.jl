#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:13:20
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * this works on gpu when using `CUDA.jl` as backend
 =#

include("../../head/cuda.jl")

@kernel function device_round(int32_x, float32_y)
    I = @index(Global)
    int32_x[I] = eltype(int32_x)(round(float32_y[I]))
end

a = rand(Int32, 10) |> CT
b = rand(Float32, 10) |> CT
device_round(Backend, 10)(a, b, ndrange = (10,))
KernelAbstractions.synchronize(Backend)
