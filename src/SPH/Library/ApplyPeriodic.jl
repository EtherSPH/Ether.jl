#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/18 22:41:24
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function sApplyPeriodic!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    ::Type{NonePeriodicBoundary},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, N, DIMENSION <: AbstractDimension{N}}
    return nothing
end

@inline function sApplyPeriodic!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    ::Type{PeriodicBoundary2D{false, false}},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, DIMENSION <: AbstractDimension{2}}
    return nothing
end

@inline function sApplyPeriodic!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    ::Type{PeriodicBoundary2D{true, false}},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, DIMENSION <: AbstractDimension{2}}
    @inbounds if @x(@i, 0) < Class.get_first_x(domain)
        @inbounds @x(@i, 0) += Class.get_span_x(domain)
    elseif @x(@i, 0) > Class.get_last_x(domain)
        @inbounds @x(@i, 0) -= Class.get_span_x(domain)
    end
    return nothing
end

@inline function sApplyPeriodic!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    ::Type{PeriodicBoundary2D{false, true}},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, DIMENSION <: AbstractDimension{2}}
    @inbounds if @x(@i, 1) < Class.get_first_y(domain)
        @inbounds @x(@i, 1) += Class.get_span_y(domain)
    elseif @x(@i, 1) > Class.get_last_y(domain)
        @inbounds @x(@i, 1) -= Class.get_span_y(domain)
    end
    return nothing
end
@inline function sApplyPeriodic!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    ::Type{PeriodicBoundary2D{true, true}},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, DIMENSION <: AbstractDimension{2}}
    sApplyPeriodic!(@self_args, domain, PeriodicBoundary2D{true, false})
    sApplyPeriodic!(@self_args, domain, PeriodicBoundary2D{false, true})
    return nothing
end

# TODO: add 3D support for apply periodic boundary
