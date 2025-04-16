#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:19:01
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractNeighbourSystemBase{IT <: Integer, CT <: AbstractArray, Backend} end

struct NeighbourSystemBase{IT <: Integer, CT <: AbstractArray, Backend} <: AbstractNeighbourSystemBase{IT, CT, Backend}
    contained_particle_index_count_::AbstractArray{IT, 1} # (n_cells, )
    contained_particle_index_list_::AbstractArray{IT, 2} # (n_cells, max_neighbour_number)
    # ! including the cell itself, this field is the only field need `atomic operation`
    neighbour_cell_index_count_::AbstractArray{IT, 1} # (n_cells, )
    neighbour_cell_index_list_::AbstractArray{IT, 2} # (n_cells, n_neighbours)
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    I::IT = @index(Global)
    i::IT, j::IT = indexLinearToCartesian(I, domain)
    n_x::IT = get_n_x(domain)
    n_y::IT = get_n_y(domain)
    for di::IT in -1:1
        ii::IT = i + di
        for dj::IT in -1:1
            jj::IT = j + dj
            if ii >= 1 && ii <= n_x && jj >= 1 && jj <= n_y
                @inbounds neighbour_cell_index_count[I] += 1
                @inbounds neighbour_cell_index_list[I, neighbour_cell_index_count[I]] =
                    indexCartesianToLinear(ii, jj, domain)
            end
        end
    end
end

@kernel function device_initializeNeighbourSystem!(
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{3}}
    I::IT = @index(Global)
    # TODO: add 3D support
end

@inline function host_initializeNeighbourSystem!(
    ::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_cell_index_count,
    neighbour_cell_index_list;
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    device_initializeNeighbourSystem!(Backend, n_threads)(
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
        ndrange = (get_n(domain),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function NeighbourSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystemBase{
    IT,
    CT,
    Backend,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, N, Dimension <: AbstractDimension{N}}
    n_cells = get_n(domain)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list = parallel(zeros(IT, n_cells, max_neighbour_number))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    neighbour_cell_count = IT(neighbourCellCount(N))
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, neighbour_cell_count))
    host_initializeNeighbourSystem!(
        parallel,
        domain,
        neighbour_cell_index_count,
        neighbour_cell_index_list;
        n_threads = n_threads,
    )
    return NeighbourSystemBase{IT, CT, Backend}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
end

function Base.show(
    io::IO,
    neighbour_system::NeighbourSystemBase{IT, CT, Backend},
) where {IT <: Integer, CT <: AbstractArray, Backend}
    println(io, "NeighbourSystemBase{IT, CT, Backend}(")
    println(io, "  number of cells: $(length(neighbour_system.contained_particle_index_count_))")
    println(io, "  max number of contained particles: $(size(neighbour_system.contained_particle_index_list_, 2))")
    println(io, "  max number of neighbour cells: $(size(neighbour_system.neighbour_cell_index_list_, 2))")
    println(io, ")")
end

# * ==================== data transfer ==================== * #

@inline function serialto!(
    destination_neighbour_system_base::AbstractNeighbourSystemBase{IT, CT1, Backend1},
    source_neighbour_system_base::AbstractNeighbourSystemBase{IT, CT2, Backend2},
)::Nothing where {IT <: Integer, CT1 <: AbstractArray, Backend1, CT2 <: AbstractArray, Backend2}
    Base.copyto!(
        destination_neighbour_system_base.contained_particle_index_count_,
        source_neighbour_system_base.contained_particle_index_count_,
    )
    Base.copyto!(
        destination_neighbour_system_base.contained_particle_index_list_,
        source_neighbour_system_base.contained_particle_index_list_,
    )
    Base.copyto!(
        destination_neighbour_system_base.neighbour_cell_index_count_,
        source_neighbour_system_base.neighbour_cell_index_count_,
    )
    Base.copyto!(
        destination_neighbour_system_base.neighbour_cell_index_list_,
        source_neighbour_system_base.neighbour_cell_index_list_,
    )
    return nothing
end

@inline function asyncto!(
    destination_neighbour_system_base::AbstractNeighbourSystemBase{IT, CT1, Backend1},
    source_neighbour_system_base::AbstractNeighbourSystemBase{IT, CT2, Backend2},
)::Nothing where {IT <: Integer, CT1 <: AbstractArray, Backend1, CT2 <: AbstractArray, Backend2}
    task1 = Threads.@spawn begin
        Base.copyto!(
            destination_neighbour_system_base.contained_particle_index_count_,
            source_neighbour_system_base.contained_particle_index_count_,
        )
        Base.copyto!(
            destination_neighbour_system_base.neighbour_cell_index_count_,
            source_neighbour_system_base.neighbour_cell_index_count_,
        )
        Base.copyto!(
            destination_neighbour_system_base.neighbour_cell_index_list_,
            source_neighbour_system_base.neighbour_cell_index_list_,
        )
    end
    task2 = Threads.@spawn begin
        Base.copyto!(
            destination_neighbour_system_base.contained_particle_index_list_,
            source_neighbour_system_base.contained_particle_index_list_,
        )
    end
    Base.fetch(task1)
    Base.fetch(task2)
    return nothing
end

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    neighbour_system_base::AbstractNeighbourSystemBase{IT, CT2, Backend2},
) where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, Backend1, CT2 <: AbstractArray, Backend2}
    n_cells = length(neighbour_system_base.contained_particle_index_count_)
    contained_particle_index_count = parallel(zeros(IT, n_cells))
    contained_particle_index_list =
        parallel(zeros(IT, n_cells, size(neighbour_system_base.contained_particle_index_list_, 2)))
    neighbour_cell_index_count = parallel(zeros(IT, n_cells))
    neighbour_cell_index_list = parallel(zeros(IT, n_cells, size(neighbour_system_base.neighbour_cell_index_list_, 2)))
    new_neighbour_system_base = NeighbourSystemBase{IT, CT1, Backend1}(
        contained_particle_index_count,
        contained_particle_index_list,
        neighbour_cell_index_count,
        neighbour_cell_index_list,
    )
    serialto!(new_neighbour_system_base, neighbour_system_base)
    return new_neighbour_system_base
end

@inline function mirror(
    neighbour_system_base::AbstractNeighbourSystemBase{IT, CT, Backend},
) where {IT <: Integer, CT <: AbstractArray, Backend}
    parallel = Environment.ParallelCPU{IT, Float32}()
    return mirror(parallel, neighbour_system_base)
end
