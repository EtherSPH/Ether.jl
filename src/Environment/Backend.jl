#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 14:27:45
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

const kNameToContainer = OrderedDict(
    "cpu" => "Array",
    "cuda" => "CUDA.CuArray",
    "rocm" => "AMDGPU.ROCArray",
    "oneapi" => "oneAPI.oneArray",
    "metal" => "Metal.MtlArray",
)

const kNameToBackend = OrderedDict(
    "cpu" => "KernelAbstractions.CPU()",
    "cuda" => "CUDA.CUDABackend()",
    "rocm" => "AMDGPU.ROCBackend()",
    "oneapi" => "oneAPI.oneAPIBackend()",
    "metal" => "Metal.MetalBackend()",
)

const kContainerToName = OrderedDict(
    "Array" => "cpu",
    "CuArray" => "cuda",
    "ROCArray" => "rocm",
    "oneArray" => "oneapi",
    "MtlArray" => "metal",
)

const kNameToColor =
    OrderedDict("cpu" => :magenta, "cuda" => :green, "rocm" => :red, "oneapi" => :blue, "metal" => :cyan)
