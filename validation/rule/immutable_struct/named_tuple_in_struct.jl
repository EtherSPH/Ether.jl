#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:03:31
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * can change
 =#

using Accessors

@kwdef struct A
    x::Vector{Int} = [1, 2, 3]
    nt::NamedTuple = (a = 1, b = 2)
end

a = A()
@reset a.nt.a = 3
println("@reset a.nt.a = 3 works")
println(a)

function reset_a_as_4!(a::A)
    @reset a.nt.a = 4
end

reset_a_as_4!(a)
println("reset_a_as_4!(a) does not work")
println(a)
