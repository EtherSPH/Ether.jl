#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:06:34
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Library

using Ether.Environment
using Ether.Math
using Ether.Macro
using Ether.SPH.Macro
using Ether.SPH.Kernel
using Ether.Class

@inline function avoidzero(r::Real, h::Real)::typeof(r)
    return r * r + typeof(r)(0.01) * h * h
end

@inline function vdotx(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
)::@float() where {N, DIMENSION <: AbstractDimension{N}}
    v_dot_x::@float() = @float 0.0
    for i::@int() in 0:(N - 1)
        @inbounds v_dot_x += @rvec(@ij, i) * (@u(@i, i) - @u(@j, i))
    end
    return v_dot_x
end

include("State.jl")
include("Kernel.jl")
include("Motion.jl")
include("Continuity.jl")
include("Pressure.jl")
include("Viscosity.jl")
include("Filter.jl")
include("ApplyPeriodic.jl")
include("Wall.jl")
include("Thermal.jl")

end # module Library
