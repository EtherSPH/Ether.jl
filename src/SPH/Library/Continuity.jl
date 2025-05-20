#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:35:12
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function sContinuity!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dt::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @rho(@i) += @drho(@i) * @float(dt)
    @inbounds @drho(@i) = @float 0.0
    return nothing
end

@inline function iDiffuse!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dw::Real = 0,
    delta::Real = 0.1,
    h::Real = 0,
    c::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @drho(@i) +=
        2 * @float(delta) * @float(h) * @float(c) * @float(dw) * @vol(@j) * (@rho(@i) - @rho(@j)) / @r(@ij)
    return nothing
end

@inline function iClassicContinuity!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dw::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @drho(@i) += @mass(@j) * vdotx(@inter_args) * @float(dw) / @r(@ij)
    return nothing
end

@inline function iBalancedContinuity!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dw::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @drho(@i) += @rho(@i) * @vol(@j) * vdotx(@inter_args) * @float(dw) / @r(@ij)
    return nothing
end
