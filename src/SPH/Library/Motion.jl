#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:31:36
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function sAccelerate!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dt::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @u(@i, i) += @du(@i, i) * @float(dt)
        @inbounds @du(@i, i) = @float 0
    end
    return nothing
end

@inline function sMove!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dt::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @x(@i, i) += @u(@i, i) * @float(dt)
    end
    return nothing
end

@inline function sAccelerateMove!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    dt::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds for i::@int() in 0:(N - 1)
        @inbounds @x(@i, i) += (@u(@i, i) + @du(@i, i) * @float(dt) * @float(0.5)) * @float(dt)
        @inbounds @u(@i, i) += @du(@i, i) * @float(dt)
        @inbounds @du(@i, i) = @float 0
    end
    return nothing
end
