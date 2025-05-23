#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/21 16:24:20
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function iClassicThermal!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dw::Real = 0,
    h::Real = 0,
    kappa::Real = 1,
    cp::Real = 1,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    heat::@float() =
        2 * @float(kappa) * @r(@ij) * @float(dw) * (@T(@i) - @T(@j)) / (@rho(@i) * avoidzero(@r(@ij), @float(h)))
    @inbounds @dT(@i) += @vol(@j) * heat / @float(cp)
    return nothing
end

@inline function sThermal!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dt::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @T(@i) += @dT(@i) * @float(dt)
    @inbounds @dT(@i) = @float 0.0
    return nothing
end
