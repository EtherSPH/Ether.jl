#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:33:31
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== AbstractDomain2D ==================== *

abstract type AbstractDomain2D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain{IT, FT, Dimension2D} end

# * ==================== Domain2D ==================== *

struct Domain2D{IT <: Integer, FT <: AbstractFloat} <: AbstractDomain2D{IT, FT}
    gap_::FT
    gap_square_::FT
    n_x_::IT
    n_y_::IT
    n_::IT
    first_x_::FT
    last_x_::FT
    first_y_::FT
    last_y_::FT
    span_x_::FT
    span_y_::FT
    gap_x_::FT
    gap_y_::FT
    gap_x_inv_::FT
    gap_y_inv_::FT
end

function Domain2D{IT, FT}(
    gap::Real,
    first_x::Real,
    first_y::Real,
    last_x::Real,
    last_y::Real,
)::Domain2D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    gap = FT(gap)
    gap_square = gap * gap
    first_x = FT(first_x)
    first_y = FT(first_y)
    last_x = FT(last_x)
    last_y = FT(last_y)
    span_x = last_x - first_x
    span_y = last_y - first_y
    n_x = floor(IT, span_x / gap)
    n_y = floor(IT, span_y / gap)
    n = n_x * n_y
    gap_x = span_x / n_x
    gap_y = span_y / n_y
    gap_x_inv = 1 / gap_x
    gap_y_inv = 1 / gap_y
    return Domain2D{IT, FT}(
        gap,
        gap_square,
        n_x,
        n_y,
        n,
        first_x,
        last_x,
        first_y,
        last_y,
        span_x,
        span_y,
        gap_x,
        gap_y,
        gap_x_inv,
        gap_y_inv,
    )
end

function Domain2D(gap::Real, first_x::Real, first_y::Real, last_x::Real, last_y::Real)
    FT = typeof(gap)
    IT = Int32
    return Domain2D{IT, FT}(gap, first_x, first_y, last_x, last_y)
end

@inline function Domain2D{IT, FT}()::Domain2D{IT, FT} where {IT <: Integer, FT <: AbstractFloat}
    return Domain2D{IT, FT}(1, 0, 0, 1, 1)
end

function Base.show(io::IO, domain::Domain2D{IT, FT}) where {IT <: Integer, FT <: AbstractFloat}
    println(io, "Domain2D{$IT, $FT}(")
    println(io, "    gap: ", get_gap(domain))
    println(io, "    n_x: ", get_n_x(domain))
    println(io, "    n_y: ", get_n_y(domain))
    println(io, "    n: ", get_n(domain))
    println(io, "    first_x: ", get_first_x(domain))
    println(io, "    last_x: ", get_last_x(domain))
    println(io, "    first_y: ", get_first_y(domain))
    println(io, "    last_y: ", get_last_y(domain))
    println(io, "    span_x: ", get_span_x(domain))
    println(io, "    span_y: ", get_span_y(domain))
    println(io, "    gap_x: ", get_gap_x(domain))
    println(io, "    gap_y: ", get_gap_y(domain))
    println(io, "    gap_x_inv: ", get_gap_x_inv(domain))
    println(io, "    gap_y_inv: ", get_gap_y_inv(domain))
    return println(io, ")")
end

# * ==================== functions ==================== *

@inline function indexCartesianToLinear(
    i::IT,
    j::IT,
    domain::AbstractDomain2D{IT, FT},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    return i + get_n_x(domain) * (j - IT(1))
end

@inline function indexLinearToCartesian(
    index::IT,
    domain::AbstractDomain{IT, FT, Dimension2D},
)::Tuple{IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    n_x = get_n_x(domain)
    i = mod1(index, n_x)
    j = cld(index, n_x)
    return i, j
end

@inline function inside(
    x::FT,
    y::FT,
    domain::AbstractDomain{IT, FT, Dimension2D},
)::Bool where {IT <: Integer, FT <: AbstractFloat}
    return (get_first_x(domain) <= x <= get_last_x(domain) && get_first_y(domain) <= y <= get_last_y(domain))
end

@inline function indexCartesianFromPosition(
    x::FT,
    y::FT,
    domain::AbstractDomain{IT, FT, Dimension2D},
)::Tuple{IT, IT} where {IT <: Integer, FT <: AbstractFloat}
    i::IT = min(get_n_x(domain), device_floor(IT, (x - get_first_x(domain)) * get_gap_x_inv(domain)) + 1)
    i = max(IT(1), i)
    j::IT = min(get_n_y(domain), device_floor(IT, (y - get_first_y(domain)) * get_gap_y_inv(domain)) + 1)
    j = max(IT(1), j)
    return i, j
end

@inline function indexLinearFromPosition(
    x::FT,
    y::FT,
    domain::AbstractDomain{IT, FT, Dimension2D},
)::IT where {IT <: Integer, FT <: AbstractFloat}
    i, j = indexCartesianFromPosition(x, y, domain)
    return indexCartesianToLinear(i, j, domain)
end
