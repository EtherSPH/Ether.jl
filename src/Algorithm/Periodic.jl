#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 21:31:09
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== periodic boundary ==================== * #

@inline function periodic(
    ::AbstractDomain{IT, FT, Dimension},
    ::Type{<:NonePeriodicBoundary},
    cell_index::IT,
    neighbour_cell_index::IT,
    dx::FT,
    dy::FT,
)::Tuple{FT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    return dx, dy
end

@inline function periodic(
    ::AbstractDomain{IT, FT, Dimension},
    ::Type{PeriodicBoundary2D{false, false}},
    cell_index::IT,
    neighbour_cell_index::IT,
    dx::FT,
    dy::FT,
)::Tuple{FT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    return dx, dy
end

# TODO: add 3D support
# TODO: add PeriodicBoundary support
