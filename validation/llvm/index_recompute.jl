#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 15:19:35
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * julia compiler will optimize the code to calculate `param.i + 1` only once.
 =#

@inline function f1!(x, param)
    x[param.i + 1] += 1
end

@inline function f2!(x, param)
    x[param.i + 1] += 2
end

@inline function f3!(x, param)
    x[param.i + 1] += 3
end

@inline function f4!(x, param)
    x[param.i + 1] += 4
end

@inline function f!(x, param)
    f1!(x, param)
    f2!(x, param)
    f3!(x, param)
    f4!(x, param)
end

x = zeros(2)
param = (i = 1,)
@code_llvm f!(x, param)
