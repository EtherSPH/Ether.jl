#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/18 18:28:47
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function host_buildPeriodic!(
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary};
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    reset!(neighbour_system)
    device_buildPeriodic!(Backend, n_threads)(
        PeriodicBoundary,
        domain,
        neighbour_system.base_.neighbour_cell_index_count_,
        neighbour_system.base_.neighbour_cell_index_list_,
        ndrange = (get_n(domain),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

# * ==================== HavePeriodicBoundary 2D ==================== * #

@inline function recordCorrection(
    ::Type{NonePeriodicBoundary},
    i1::IT,
    j1::IT,
    i2::IT,
    j2::IT,
    n_x::IT,
    n_y::IT,
)::Tuple{Bool, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2
    else
        return false, i2, j2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary2D{false, false}},
    i1::IT,
    j1::IT,
    i2::IT,
    j2::IT,
    n_x::IT,
    n_y::IT,
)::Tuple{Bool, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2
    else
        return false, i2, j2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary2D{true, false}},
    i1::IT,
    j1::IT,
    i2::IT,
    j2::IT,
    n_x::IT,
    n_y::IT,
)::Tuple{Bool, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2
    elseif i1 == 1 && i2 == 0 && j2 >= 1 && j2 <= n_y
        return true, n_x, j2
    elseif i1 == n_x && i2 == n_x + 1 && j2 >= 1 && j2 <= n_y
        return true, 1, j2
    else
        return false, i2, j2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary2D{false, true}},
    i1::IT,
    j1::IT,
    i2::IT,
    j2::IT,
    n_x::IT,
    n_y::IT,
)::Tuple{Bool, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2
    elseif j1 == 1 && j2 == 0 && i2 >= 1 && i2 <= n_x
        return true, i2, n_y
    elseif j1 == n_y && j2 == n_y + 1 && i2 >= 1 && i2 <= n_x
        return true, 1, j2
    else
        return false, i2, j2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary2D{true, true}},
    i1::IT,
    j1::IT,
    i2::IT,
    j2::IT,
    n_x::IT,
    n_y::IT,
)::Tuple{Bool, IT, IT} where {IT <: Integer}
    if i2 == 0
        i2 = n_x
    elseif i2 == n_x + 1
        i2 = 1
    end
    if j2 == 0
        j2 = n_y
    elseif j2 == n_y + 1
        j2 = 1
    end
    return true, i2, j2
end

@kernel function device_buildPeriodic!(
    ::Type{PeriodicBoundary},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    Dimension <: AbstractDimension{2},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    I::IT = @index(Global)
    i::IT, j::IT = indexLinearToCartesian(I, domain)
    n_x::IT = get_n_x(domain)
    n_y::IT = get_n_y(domain)
    for di::IT in -1:1
        ii::IT = i + di
        for dj::IT in -1:1
            jj::IT = j + dj
            record, ii, jj = recordCorrection(PeriodicBoundary, i, j, ii, jj, n_x, n_y)
            if record == true
                @inbounds neighbour_cell_index_count[I] += 1
                @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                    indexCartesianToLinear(ii, jj, domain)
            end
        end
    end
end

# * ==================== HavePeriodicBoundary 3D ==================== * #

@inline function recordCorrection(
    ::Type{NonePeriodicBoundary},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, i2, j2, k2
    else
        return false, i2, j2, k2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{false, false, false}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, i2, j2, k2
    else
        return false, i2, j2, k2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{true, false, false}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, i2, j2, k2
    elseif i1 == 1 && i2 == 0 && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, n_x, j2, k2
    elseif i1 == n_x && i2 == n_x + 1 && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, 1, j2, k2
    else
        return false, i2, j2, k2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{false, true, false}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, i2, j2, k2
    elseif j1 == 1 && j2 == 0 && i2 >= 1 && i2 <= n_x && k2 >= 1 && k2 <= n_z
        return true, i2, n_y, k2
    elseif j1 == n_y && j2 == n_y + 1 && i2 >= 1 && i2 <= n_x && k2 >= 1 && k2 <= n_z
        return true, i2, 1, k2
    else
        return false, i2, j2, k2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{false, false, true}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    if i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y && k2 >= 1 && k2 <= n_z
        return true, i2, j2, k2
    elseif k1 == 1 && k2 == 0 && i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2, n_z
    elseif k1 == n_z && k2 == n_z + 1 && i2 >= 1 && i2 <= n_x && j2 >= 1 && j2 <= n_y
        return true, i2, j2, 1
    else
        return false, i2, j2, k2
    end
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{true, true, false}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    # TODO: add 3D support for PeriodicBoundary3D{true, true, false}
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{false, true, true}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    # TODO: add 3D support for PeriodicBoundary3D{false, true, true}
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{true, false, true}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    # TODO: add 3D support for PeriodicBoundary3D{true, false, true}
end

@inline function recordCorrection(
    ::Type{PeriodicBoundary3D{true, true, true}},
    i1::IT,
    j1::IT,
    k1::IT,
    i2::IT,
    j2::IT,
    k2::IT,
    n_x::IT,
    n_y::IT,
    n_z::IT,
)::Tuple{Bool, IT, IT, IT} where {IT <: Integer}
    # TODO: add 3D support for PeriodicBoundary3D{true, true, true}
end

@kernel function device_buildPeriodic!(
    ::Type{PeriodicBoundary},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    Dimension <: AbstractDimension{3},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    I::IT = @index(Global)
    i::IT, j::IT, k::IT = indexLinearToCartesian(I, domain)
    n_x::IT = get_n_x(domain)
    n_y::IT = get_n_y(domain)
    n_z::IT = get_n_z(domain)
    for di::IT in -1:1
        ii::IT = i + di
        for dj::IT in -1:1
            jj::IT = j + dj
            for dk::IT in -1:1
                kk::IT = k + dk
                record, ii, jj, kk = recordCorrection(PeriodicBoundary, i, j, k, ii, jj, kk, n_x, n_y, n_z)
                if record == true
                    @inbounds neighbour_cell_index_count[I] += 1
                    @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                        indexCartesianToLinear(ii, jj, kk, domain)
                end
            end
        end
    end
end
