#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/13 17:14:02
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module DataIO

using JLD2
using CodecZstd
using OrderedCollections
using JSON
using YAML
using KernelAbstractions

using Ether.Utility
using Ether.Environment
using Ether.Class

include("Writer.jl")
include("Config.jl")

export AbstractWriter

end # module DataIO
