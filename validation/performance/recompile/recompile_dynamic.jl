#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:44:07
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * causing no recompilation as the `length(a)` varies
 =#


 include("../../head/oneapi.jl")

 @kernel function device_add!(a, b)
     I = @index(Global)
     a[I] += b[I]
 end
 
 @inline function host_add!(a, b)
     kernel_add! = device_add!(Backend, 256)(a, b, ndrange = (length(a),))
     KernelAbstractions.synchronize(Backend)
 end
 
 a1 = randn(FT, 1024) |> CT
 b1 = randn(FT, 1024) |> CT
 a2 = randn(FT, 512) |> CT
 b2 = randn(FT, 512) |> CT
 
 @time host_add!(a1, b1)
 @time host_add!(a1, b1)
 @time host_add!(a2, b2)
 @time host_add!(a2, b2)
 
 # causing no recompilation as the `length(a)` varies
 # 0.566049 seconds (372.60 k allocations: 18.464 MiB, 71.57% compilation time)
 # 0.000885 seconds (67 allocations: 2.688 KiB)
 # 0.000934 seconds (67 allocations: 2.688 KiB)
 # 0.002592 seconds (67 allocations: 2.688 KiB)