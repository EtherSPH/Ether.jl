#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:16:43
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function neighbourCellCount(dim::IT)::IT where {IT <: Integer}
    return 3^dim # 9 for 2D, 27 for 3D
end

include("NeighbourSystemBase.jl")
include("ActivePair.jl")
include("PeriodicBoundary.jl")

abstract type AbstractNeighbourSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundaryPolicy <: AbstractPeriodicBoundary,
} end

struct NeighbourSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
} <: AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary}
    base_::NeighbourSystemBase{IT, CT, Backend}
    active_pair_::ActivePair{IT, CT, Backend}
end

@inline function NeighbourSystem(
    ::Type{PeriodicBoundary},
    parallel::AbstractParallel{IT, FT, CT, Backend},
    domain::AbstractDomain{IT, FT, Dimension},
    active_pair::AbstractVector{<:Pair{<:Integer, <:Integer}};
    max_neighbour_number::Integer = kDefaultMaxNeighbourNumber,
    n_threads::Integer = kDefaultThreadNumber,
)::NeighbourSystem{
    IT,
    FT,
    CT,
    Backend,
    Dimension,
    PeriodicBoundary,
} where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    base = NeighbourSystemBase(parallel, domain; max_neighbour_number = max_neighbour_number, n_threads = n_threads)
    _active_pair = ActivePair(parallel, active_pair)
    return NeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary}(base, _active_pair)
end

function Base.show(
    io::IO,
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    println(io, "Ether.NeighbourSystem{$IT, $FT, $CT, $Backend, $Dimension, $PeriodicBoundary}(")
    println(io, "  base:")
    println(io, "    number of cells: $(get_n_cells(neighbour_system))")
    println(
        io,
        "    max number of contained particles: $(size(neighbour_system.base_.contained_particle_index_count_, 2))",
    )
    println(io, "    max number of neighbour cells: $(size(neighbour_system.base_.neighbour_cell_index_list_, 2))")
    println(io, "  active_pair:")
    println(io, "    pair_vector: $(neighbour_system.active_pair_.pair_vector_)")
    println(io, "    adjacency_matrix: $(neighbour_system.active_pair_.adjacency_matrix_)")
    println(io, ")")
end

# * ==================== AbstractNeighbourSystem function ========================== * #

@inline function get_n_cells(
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
)::IT where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    return length(neighbour_system.base_.contained_particle_index_count_)
end

@inline function clean!(
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    KernelAbstractions.fill!(neighbour_system.base_.contained_particle_index_count_, IT(0))
    return nothing
end

# * ==================== NeighbourSystem data transfer function ========================== * #

@inline function serialto!(
    destination_neighbour_system::AbstractNeighbourSystem{IT, FT, CT1, Backend1, Dimension, PeriodicBoundary},
    source_neighbour_system::AbstractNeighbourSystem{IT, FT, CT2, Backend2, Dimension, PeriodicBoundary},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    Backend1,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
    CT2 <: AbstractArray,
    Backend2,
}
    serialto!(destination_neighbour_system.base_, source_neighbour_system.base_)
    serialto!(destination_neighbour_system.active_pair_, source_neighbour_system.active_pair_)
    return nothing
end

@inline function asyncto!(
    destination_neighbour_system::AbstractNeighbourSystem{IT, FT, CT1, Backend1, Dimension, PeriodicBoundary},
    source_neighbour_system::AbstractNeighbourSystem{IT, FT, CT2, Backend2, Dimension, PeriodicBoundary},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    Backend1,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
    CT2 <: AbstractArray,
    Backend2,
}
    task1 = Threads.@spawn asyncto!(destination_neighbour_system.base_, source_neighbour_system.base_)
    task2 = Threads.@spawn asyncto!(destination_neighbour_system.active_pair_, source_neighbour_system.active_pair_)
    Base.fetch(task1)
    Base.fetch(task2)
    return nothing
end

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT2, Backend2, Dimension, PeriodicBoundary},
)::AbstractNeighbourSystem{
    IT,
    FT,
    CT1,
    Backend1,
    Dimension,
    PeriodicBoundary,
} where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    Backend1,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
    CT2 <: AbstractArray,
    Backend2,
}
    return NeighbourSystem{IT, FT, CT1, Backend1, Dimension, PeriodicBoundary}(
        mirror(parallel, neighbour_system.base_),
        mirror(parallel, neighbour_system.active_pair_),
    )
end

@inline function mirror(
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
) where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    parallel = Environment.ParallelCPU{IT, FT}()
    return mirror(parallel, neighbour_system)
end
