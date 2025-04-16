#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:36:23
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module SPH

include("Kernel/Kernel.jl")
include("Macro/Macro.jl")
include("Library/Library.jl")

end # module SPH
