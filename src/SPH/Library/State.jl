#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:18:50
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function sVolume!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @vol(@i) = @mass(@i) / @rho(@i)
    return nothing
end

@inline function sAcceleration!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    # copy `du` to `a`
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @a(@i, i) = @du(@i, i)
    end
    return nothing
end

@inline function sGravity!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    gx::Real = 0,
    gy::Real = 0,
)::Nothing where {DIMENSION <: AbstractDimension{2}}
    @inbounds @du(@i, 0) += @float gx
    @inbounds @du(@i, 1) += @float gy
    return nothing
    return nothing
end

@inline function sGravity!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    gx::Real = 0,
    gy::Real = 0,
    gz::Real = 0,
)::Nothing where {DIMENSION <: AbstractDimension{3}}
    @inbounds @du(@i, 0) += @float gx
    @inbounds @du(@i, 1) += @float gy
    @inbounds @du(@i, 2) += @float gz
    return nothing
end
