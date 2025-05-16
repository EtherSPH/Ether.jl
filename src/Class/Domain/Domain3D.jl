#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:33:38
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== AbstractDomain2D ==================== *

abstract type AbstractDomain3D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension3D} end

@inline function get_n_z(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.n_z_
end

@inline function get_first_z(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.first_z_
end

@inline function get_last_z(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.last_z_
end

@inline function get_span_z(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.span_z_
end

@inline function get_gap_z(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.gap_z_
end

@inline function get_gap_z_inv(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.gap_z_inv_
end

@inline function get_n_xy(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.n_xy_
end

@inline function get_n_yz(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.n_yz_
end

@inline function get_n_zx(domain::AbstractDomain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    return domain.n_zx_
end

# * ==================== Domain3D ==================== *

struct Domain3D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain3D{IT, FT}
    gap_::FT
    gap_square_::FT
    n_x_::IT
    n_y_::IT
    n_z_::IT
    n_xy_::IT
    n_yz_::IT
    n_zx_::IT
    n_::IT
    first_x_::FT
    last_x_::FT
    first_y_::FT
    last_y_::FT
    first_z_::FT
    last_z_::FT
    span_x_::FT
    span_y_::FT
    span_z_::FT
    gap_x_::FT
    gap_y_::FT
    gap_z_::FT
    gap_x_inv_::FT
    gap_y_inv_::FT
    gap_z_inv_::FT
end

function Domain3D{IT, FT}(
    gap::Real,
    first_x::Real,
    first_y::Real,
    first_z::Real,
    last_x::Real,
    last_y::Real,
    last_z::Real,
)::Domain3D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    gap = FT(gap)
    gap_square = gap * gap
    first_x = FT(first_x)
    first_y = FT(first_y)
    first_z = FT(first_z)
    last_x = FT(last_x)
    last_y = FT(last_y)
    last_z = FT(last_z)
    span_x = last_x - first_x
    span_y = last_y - first_y
    span_z = last_z - first_z
    n_x = floor(IT, span_x / gap)
    n_y = floor(IT, span_y / gap)
    n_z = floor(IT, span_z / gap)
    n_xy = n_x * n_y
    n_yz = n_y * n_z
    n_zx = n_z * n_x
    n = n_x * n_y * n_z
    gap_x = span_x / n_x
    gap_y = span_y / n_y
    gap_z = span_z / n_z
    gap_x_inv = 1 / gap_x
    gap_y_inv = 1 / gap_y
    gap_z_inv = 1 / gap_z
    return Domain3D{IT, FT}(
        gap,
        gap_square,
        n_x,
        n_y,
        n_z,
        n_xy,
        n_yz,
        n_zx,
        n,
        first_x,
        last_x,
        first_y,
        last_y,
        first_z,
        last_z,
        span_x,
        span_y,
        span_z,
        gap_x,
        gap_y,
        gap_z,
        gap_x_inv,
        gap_y_inv,
        gap_z_inv,
    )
end

function Domain3D(gap::Real, first_x::Real, first_y::Real, first_z::Real, last_x::Real, last_y::Real, last_z::Real)
    FT = typeof(gap)
    IT = Int32
    return Domain3D{IT, FT}(gap, first_x, first_y, first_z, last_x, last_y, last_z)
end

@inline function Domain3D{IT, FT}()::Domain2D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    return Domain3D{IT, FT}(1, 0, 0, 0, 1, 1, 1)
end

function Base.show(io::IO, domain::Domain3D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    println(io, "Domain2D{$IT, $FT}(")
    println(io, "    gap: ", get_gap(domain))
    println(io, "    n_x: ", get_n_x(domain))
    println(io, "    n_y: ", get_n_y(domain))
    println(io, "    n_z: ", get_n_z(domain))
    println(io, "    n_xy: ", get_n_xy(domain))
    println(io, "    n_yz: ", get_n_yz(domain))
    println(io, "    n_zx: ", get_n_zx(domain))
    println(io, "    n: ", get_n(domain))
    println(io, "    first_x: ", get_first_x(domain))
    println(io, "    last_x: ", get_last_x(domain))
    println(io, "    first_y: ", get_first_y(domain))
    println(io, "    last_y: ", get_last_y(domain))
    println(io, "    first_z: ", get_first_z(domain))
    println(io, "    last_z: ", get_last_z(domain))
    println(io, "    span_x: ", get_span_x(domain))
    println(io, "    span_y: ", get_span_y(domain))
    println(io, "    span_z: ", get_span_z(domain))
    println(io, "    gap_x: ", get_gap_x(domain))
    println(io, "    gap_y: ", get_gap_y(domain))
    println(io, "    gap_z: ", get_gap_z(domain))
    println(io, "    gap_x_inv: ", get_gap_x_inv(domain))
    println(io, "    gap_y_inv: ", get_gap_y_inv(domain))
    println(io, "    gap_z_inv: ", get_gap_z_inv(domain))
    return println(io, ")")
end

# * ==================== functions ==================== *

@inline function indexCartesianToLinear(
    i::IT,
    j::IT,
    k::IT,
    domain::AbstractDomain3D{IT, FT},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    return i + get_n_x(domain) * (j - IT(1)) + get_n_xy(domain) * (k - IT(1))
end

@inline function indexLinearToCartesian(
    index::IT,
    domain::AbstractDomain3D{IT, FT},
)::Tuple{IT, IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    n_x = get_n_x(domain)
    n_xy = get_n_xy(domain)
    i = mod1(index, n_x)
    k = cld(index, n_xy)
    index -= (k - IT(1)) * n_xy + i
    j = cld(index, n_x) + IT(1)
    return i, j, k
end

@inline function inside(
    x::FT,
    y::FT,
    z::FT,
    domain::AbstractDomain3D{IT, FT},
)::Bool where {IT <: Integer, FT <: AbstractFloat}
    return (x >= get_first_x(domain)) &&
           (x <= get_last_x(domain)) &&
           (y >= get_first_y(domain)) &&
           (y <= get_last_y(domain)) &&
           (z >= get_first_z(domain)) &&
           (z <= get_last_z(domain))
end

@inline function indexCartesianFromPosition(
    x::FT,
    y::FT,
    z::FT,
    domain::AbstractDomain3D{IT, FT},
)::Tuple{IT, IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    i::IT = min(get_n_x(domain), device_floor(IT, (x - get_first_x(domain)) * get_gap_x_inv(domain)) + 1)
    i = max(IT(1), i)
    j::IT = min(get_n_y(domain), device_floor(IT, (y - get_first_y(domain)) * get_gap_y_inv(domain)) + 1)
    j = max(IT(1), j)
    k::IT = min(get_n_z(domain), device_floor(IT, (z - get_first_z(domain)) * get_gap_z_inv(domain)) + 1)
    k = max(IT(1), k)
    return i, j, k
end

@inline function indexLinearFromPosition(
    x::FT,
    y::FT,
    z::FT,
    domain::AbstractDomain3D{IT, FT},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    i, j, k = indexCartesianFromPosition(x, y, z, domain)
    return indexCartesianToLinear(i, j, k, domain)
end
