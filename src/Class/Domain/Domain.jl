#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:28:01
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractDomain{IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension} end

@inline function device_floor(IT::Type{<:Integer}, x::FT)::IT where {FT <: AbstractFloat}
    # floor function used on device
    # why here is `unsafe_trunc`?
    # this problem quite annoys me during the whole 2025 year's Spring Festival
    # luckily, I found the solution in the issue: see [link](https://github.com/JuliaGPU/oneAPI.jl/issues/441)
    return unsafe_trunc(IT, floor(x))
end

# * ==================== AbstractDomain ==================== * #

@inline function dimension(
    ::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return IT(N)
end

@inline function get_gap(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_
end

@inline function get_gap_square(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_square_
end

@inline function get_n_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.n_x_
end

@inline function get_n_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.n_y_
end

@inline function get_n_z(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.n_z_
end

@inline function get_n(
    domain::AbstractDomain{IT, FT, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.n_
end

@inline function get_first_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.first_x_
end

@inline function get_last_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.last_x_
end

@inline function get_first_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.first_y_
end

@inline function get_last_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.last_y_
end

@inline function get_first_z(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.first_z_
end

@inline function get_last_z(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.last_z_
end

@inline function get_span_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.span_x_
end

@inline function get_span_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.span_y_
end

@inline function get_span_z(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.span_z_
end

@inline function get_gap_x(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_x_
end

@inline function get_gap_y(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_y_
end

@inline function get_gap_z(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_z_
end

@inline function get_gap_x_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_x_inv_
end

@inline function get_gap_y_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_y_inv_
end

@inline function get_gap_z_inv(
    domain::AbstractDomain{IT, FT, Dimension},
)::FT where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    return domain.gap_z_inv_
end

include("Domain2D.jl")
include("Domain3D.jl")
