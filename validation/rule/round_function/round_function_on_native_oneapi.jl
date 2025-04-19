#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:14:35
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # ! no, not working
 =#

include("../../head/oneapi.jl")

function device_round(int32_x, float32_y)
    I = get_global_id()
    round_float32_y::Float32 = round(float32_y[I])
    round_int32_y::Int32 = Int32(round_float32_y)
    int32_x[I] = round_int32_y
    return
end

a = rand(Int32, 10) |> CT
b = rand(Float32, 10) |> CT
@oneapi items = 10 device_round(a, b)
oneAPI.synchronize()
