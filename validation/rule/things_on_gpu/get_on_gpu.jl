#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:24:17
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * getfield and getproperty are allowed on device!
 =#

 include("../../head/oneapi.jl")

 @kernel function device_f!(x, nt::NamedTuple)
     I = @index(Global)
     x[I, 1] += getfield(nt, :a) # allowed
     x[I, 2] += getproperty(nt, :b) # allowed
 end
 
 a = zeros(Float32, 3, 2) |> CT
 nt = (a = 1.0f0, b = 2.0f0)
 device_f!(Backend, 256)(a, nt, ndrange = (3,))
 KernelAbstractions.synchronize(Backend)
 @info a