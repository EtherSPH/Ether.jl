#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 21:31:09
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== periodic boundary 2D ==================== * #

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

@inline function periodic(
    domain::AbstractDomain{IT, FT, Dimension},
    ::Type{PeriodicBoundary2D{true, false}},
    cell_index::IT,
    neighbour_cell_index::IT,
    dx::FT,
    dy::FT,
)::Tuple{FT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    i1, _ = Class.indexLinearToCartesian(cell_index, domain)
    i2, _ = Class.indexLinearToCartesian(neighbour_cell_index, domain)
    if i2 > 1 && i2 < Class.get_n_x(domain)
        return dx, dy
    elseif i1 == 1 && i2 == Class.get_n_x(domain)
        dx += Class.get_span_x(domain)
        return dx, dy
    elseif i1 == Class.get_n_x(domain) && i2 == 1
        dx -= Class.get_span_x(domain)
        return dx, dy
    else
        return dx, dy
    end
end

@inline function periodic(
    domain::AbstractDomain{IT, FT, Dimension},
    ::Type{PeriodicBoundary2D{false, true}},
    cell_index::IT,
    neighbour_cell_index::IT,
    dx::FT,
    dy::FT,
)::Tuple{FT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    _, j1 = Class.indexLinearToCartesian(cell_index, domain)
    _, j2 = Class.indexLinearToCartesian(neighbour_cell_index, domain)
    if j2 > 1 && j2 < Class.get_n_y(domain)
        return dx, dy
    elseif j1 == 1 && j2 == Class.get_n_y(domain)
        dy += Class.get_span_y(domain)
        return dx, dy
    elseif j1 == Class.get_n_y(domain) && j2 == 1
        dy -= Class.get_span_y(domain)
        return dx, dy
    else
        return dx, dy
    end
end

@inline function periodic(
    domain::AbstractDomain{IT, FT, Dimension},
    ::Type{PeriodicBoundary2D{true, true}},
    cell_index::IT,
    neighbour_cell_index::IT,
    dx::FT,
    dy::FT,
)::Tuple{FT, FT} where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    i1, j1 = Class.indexLinearToCartesian(cell_index, domain)
    i2, j2 = Class.indexLinearToCartesian(neighbour_cell_index, domain)
    if i2 > 1 && i2 < Class.get_n_x(domain) && j2 > 1 && j2 < Class.get_n_y(domain)
        return dx, dy
    end
    if i1 == 1 && i2 == Class.get_n_x(domain)
        dx += Class.get_span_x(domain)
    elseif i1 == Class.get_n_x(domain) && i2 == 1
        dx -= Class.get_span_x(domain)
    end
    if j1 == 1 && j2 == Class.get_n_y(domain)
        dy += Class.get_span_y(domain)
    elseif j1 == Class.get_n_y(domain) && j2 == 1
        dy -= Class.get_span_y(domain)
    end
    return dx, dy
end

# * ==================== periodic boundary 3D ==================== * #

# TODO: add 3D support
