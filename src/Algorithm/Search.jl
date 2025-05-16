#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 16:18:53
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== insertParticlesIntoCells ==================== * #

@kernel function device_insertParticlesIntoCells!(
    domain::AbstractDomain{IT, FT, Dimension},
    ps_is_alive,
    ps_cell_index,
    @Const(INT),
    @Const(FLOAT),
    ns_contained_particle_index_count,
    ns_contained_particle_index_list,
    index_IsMovable::IT,
    index_PositionVec::IT,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        # * case here:
        # * 1. movable: cell index must be calculated again
        # * 2. immovable: if cell_index == 0, then calculate cell index
        # *               if cell_index != 0, cell index does not need to be calculated
        @inbounds cell_index::IT = ps_cell_index[I]
        @inbounds if INT[I, index_IsMovable] == 0 && cell_index != 0
            @inbounds particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
            @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
        else
            @inbounds x::FT = FLOAT[I, index_PositionVec]
            @inbounds y::FT = FLOAT[I, index_PositionVec + 1]
            if Class.inside(x, y, domain)
                cell_index = Class.indexLinearFromPosition(x, y, domain)
                @inbounds ps_cell_index[I] = cell_index
                particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
                @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
            else
                @inbounds ps_is_alive[I] = 0
                @inbounds ps_cell_index[I] = 0
            end
        end
    end
end

@kernel function device_insertParticlesIntoCells!(
    domain::AbstractDomain{IT, FT, Dimension},
    ps_is_alive,
    ps_cell_index,
    @Const(INT),
    @Const(FLOAT),
    ns_contained_particle_index_count,
    ns_contained_particle_index_list,
    index_IsMovable::IT,
    index_PositionVec::IT,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{3}}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        # * case here:
        # * 1. movable: cell index must be calculated again
        # * 2. immovable: if cell_index == 0, then calculate cell index
        # *               if cell_index != 0, cell index does not need to be calculated
        @inbounds cell_index::IT = ps_cell_index[I]
        @inbounds if INT[I, index_IsMovable] == 0 && cell_index != 0
            @inbounds particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
            @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
        else
            @inbounds x::FT = FLOAT[I, index_PositionVec]
            @inbounds y::FT = FLOAT[I, index_PositionVec + 1]
            @inbounds z::FT = FLOAT[I, index_PositionVec + 2]
            if Class.inside(x, y, z, domain)
                cell_index = Class.indexLinearFromPosition(x, y, z, domain)
                @inbounds ps_cell_index[I] = cell_index
                particle_in_cell_index = Atomix.@atomic ns_contained_particle_index_count[cell_index] += 1
                @inbounds ns_contained_particle_index_list[cell_index, particle_in_cell_index] = I
            else
                @inbounds ps_is_alive[I] = 0
                @inbounds ps_cell_index[I] = 0
            end
        end
    end
end

@inline function host_insertParticlesIntoCells!(
    ::Static,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
    n_particles::IT,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    kernel_insertParticlesIntoCells! = device_insertParticlesIntoCells!(Backend, n_threads, (Int64(n_particles),))
    kernel_insertParticlesIntoCells!(
        domain,
        particle_system.base_.is_alive_,
        particle_system.base_.cell_index_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        particle_system.named_index_.index_.IsMovable,
        particle_system.named_index_.index_.PositionVec,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function host_insertParticlesIntoCells!(
    ::Dynamic,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
    n_particles::IT,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    device_insertParticlesIntoCells!(Backend, n_threads)(
        domain,
        particle_system.base_.is_alive_,
        particle_system.base_.cell_index_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        particle_system.named_index_.index_.IsMovable,
        particle_system.named_index_.index_.PositionVec,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
end

# * ==================== findNeighbourParticlesFromCells ==================== * #

@inline function defaultCriterion(
    ::Type{DIMENSION},
    I::Integer,
    J::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    dr_square::FT,
)::Bool where {IT <: Integer, FT <: AbstractFloat, N, DIMENSION <: AbstractDimension{N}}
    return dr_square <= Class.get_gap_square(domain)
end

@inline function symmetryCriterion(
    ::Type{DIMENSION},
    I::Integer,
    J::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    domain::AbstractDomain{IT, FT, DIMENSION},
    dr_square::FT,
)::Bool where {IT <: Integer, FT <: AbstractFloat, N, DIMENSION <: AbstractDimension{N}}
    @inbounds h2 = Math.Mean.harmonic(FLOAT[I, INDEX.H], FLOAT[J, INDEX.H]) * 2
    return dr_square <= h2 * h2
end

@kernel function device_findNeighbourParticlesFromCells!(
    domain::AbstractDomain{IT, FT, Dimension},
    periodic_boundary::Type{<:AbstractPeriodicBoundary},
    @Const(ps_is_alive),
    @Const(ps_cell_index),
    INT, # 2D array
    FLOAT, # 2D array
    ns_contained_particle_index_count, # 1D array
    ns_contained_particle_index_list, # 2D array
    ns_neighbour_cell_index_count, # 1D array
    ns_neighbour_cell_index_list, # 2D array
    ns_adjacency_matrix, # 2D array
    index_Tag::IT,
    index_PositionVec::IT,
    index_nCount::IT,
    index_nIndex::IT,
    index_nRVec::IT,
    index_nR::IT,
    INDEX::NamedTuple,
    criterion::Function,
    action!::Function,
    PARAMETER,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{2}}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        @inbounds INT[I, index_nCount] = 0
        @inbounds cell_index::IT = ps_cell_index[I]
        @inbounds n_neighbour_cells::IT = ns_neighbour_cell_index_count[cell_index]
        for i_neighbour_cell::IT in 1:n_neighbour_cells
            @inbounds neighbour_cell_index::IT = ns_neighbour_cell_index_list[cell_index, i_neighbour_cell]
            @inbounds n_particles_in_neighbour_cell::IT = ns_contained_particle_index_count[neighbour_cell_index]
            for i_particle::IT in 1:n_particles_in_neighbour_cell
                @inbounds J::IT = ns_contained_particle_index_list[neighbour_cell_index, i_particle]
                @inbounds if I != J && ns_adjacency_matrix[INT[I, index_Tag], INT[J, index_Tag]] == 1
                    @inbounds dx::FT = FLOAT[I, index_PositionVec] - FLOAT[J, index_PositionVec]
                    @inbounds dy::FT = FLOAT[I, index_PositionVec + 1] - FLOAT[J, index_PositionVec + 1]
                    dx, dy = periodic(domain, periodic_boundary, cell_index, neighbour_cell_index, dx, dy)
                    dr_square::FT = dx * dx + dy * dy
                    if criterion(Dimension, I, J, INT, FLOAT, INDEX, PARAMETER, domain, dr_square) == true
                        @inbounds INT[I, index_nCount] += 1
                        @inbounds neighbour_count::IT = INT[I, index_nCount]
                        @inbounds INT[I, index_nIndex + neighbour_count - 1] = J
                        index_dr_vec::IT = index_nRVec + 2 * (neighbour_count - 1)
                        @inbounds FLOAT[I, index_dr_vec + 0] = dx
                        @inbounds FLOAT[I, index_dr_vec + 1] = dy
                        @inbounds FLOAT[I, index_nR + neighbour_count - 1] = sqrt(dr_square)
                    end
                end
            end
        end
        action!(Dimension, I, INT, FLOAT, INDEX, PARAMETER)
    end
end

@kernel function device_findNeighbourParticlesFromCells!(
    domain::AbstractDomain{IT, FT, Dimension},
    periodic_boundary::Type{<:AbstractPeriodicBoundary},
    @Const(ps_is_alive),
    @Const(ps_cell_index),
    INT, # 2D array
    FLOAT, # 2D array
    ns_contained_particle_index_count, # 1D array
    ns_contained_particle_index_list, # 2D array
    ns_neighbour_cell_index_count, # 1D array
    ns_neighbour_cell_index_list, # 2D array
    ns_adjacency_matrix, # 2D array
    index_Tag::IT,
    index_PositionVec::IT,
    index_nCount::IT,
    index_nIndex::IT,
    index_nRVec::IT,
    index_nR::IT,
    INDEX::NamedTuple,
    criterion::Function,
    action!::Function,
    PARAMETER,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension{3}}
    I::IT = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        @inbounds INT[I, index_nCount] = 0
        @inbounds cell_index::IT = ps_cell_index[I]
        @inbounds n_neighbour_cells::IT = ns_neighbour_cell_index_count[cell_index]
        for i_neighbour_cell::IT in 1:n_neighbour_cells
            @inbounds neighbour_cell_index::IT = ns_neighbour_cell_index_list[cell_index, i_neighbour_cell]
            @inbounds n_particles_in_neighbour_cell::IT = ns_contained_particle_index_count[neighbour_cell_index]
            for i_particle::IT in 1:n_particles_in_neighbour_cell
                @inbounds J::IT = ns_contained_particle_index_list[neighbour_cell_index, i_particle]
                @inbounds if I != J && ns_adjacency_matrix[INT[I, index_Tag], INT[J, index_Tag]] == 1
                    @inbounds dx::FT = FLOAT[I, index_PositionVec] - FLOAT[J, index_PositionVec]
                    @inbounds dy::FT = FLOAT[I, index_PositionVec + 1] - FLOAT[J, index_PositionVec + 1]
                    @inbounds dz::FT = FLOAT[I, index_PositionVec + 2] - FLOAT[J, index_PositionVec + 2]
                    dx, dy, dz = periodic(domain, periodic_boundary, cell_index, neighbour_cell_index, dx, dy, dz)
                    dr_square::FT = dx * dx + dy * dy + dz * dz
                    if criterion(Dimension, I, J, INT, FLOAT, INDEX, PARAMETER, domain, dr_square) == true
                        @inbounds INT[I, index_nCount] += 1
                        @inbounds neighbour_count::IT = INT[I, index_nCount]
                        @inbounds INT[I, index_nIndex + neighbour_count - 1] = J
                        index_dr_vec::IT = index_nRVec + 3 * (neighbour_count - 1)
                        @inbounds FLOAT[I, index_dr_vec + 0] = dx
                        @inbounds FLOAT[I, index_dr_vec + 1] = dy
                        @inbounds FLOAT[I, index_dr_vec + 2] = dz
                        @inbounds FLOAT[I, index_nR + neighbour_count - 1] = sqrt(dr_square)
                    end
                end
            end
        end
        action!(Dimension, I, INT, FLOAT, INDEX, PARAMETER)
    end
end

@inline function host_findNeighbourParticlesFromCells!(
    ::Static,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
    parameter,
    n_particles::IT,
    criterion::Function,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    kernel_findNeighbourParticlesFromCells! =
        device_findNeighbourParticlesFromCells!(Backend, n_threads, (Int64(n_particles),))
    kernel_findNeighbourParticlesFromCells!(
        domain,
        PeriodicBoundary,
        particle_system.base_.is_alive_,
        particle_system.base_.cell_index_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        neighbour_system.base_.neighbour_cell_index_count_,
        neighbour_system.base_.neighbour_cell_index_list_,
        neighbour_system.active_pair_.adjacency_matrix_,
        particle_system.named_index_.index_.Tag,
        particle_system.named_index_.index_.PositionVec,
        particle_system.named_index_.index_.nCount,
        particle_system.named_index_.index_.nIndex,
        particle_system.named_index_.index_.nRVec,
        particle_system.named_index_.index_.nR,
        particle_system.named_index_.index_,
        criterion,
        action!,
        parameter,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function host_findNeighbourParticlesFromCells!(
    ::Dynamic,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
    parameter,
    n_particles::IT,
    criterion::Function,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    device_findNeighbourParticlesFromCells!(Backend, n_threads)(
        domain,
        PeriodicBoundary,
        particle_system.base_.is_alive_,
        particle_system.base_.cell_index_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        neighbour_system.base_.contained_particle_index_count_,
        neighbour_system.base_.contained_particle_index_list_,
        neighbour_system.base_.neighbour_cell_index_count_,
        neighbour_system.base_.neighbour_cell_index_list_,
        neighbour_system.active_pair_.adjacency_matrix_,
        particle_system.named_index_.index_.Tag,
        particle_system.named_index_.index_.PositionVec,
        particle_system.named_index_.index_.nCount,
        particle_system.named_index_.index_.nIndex,
        particle_system.named_index_.index_.nRVec,
        particle_system.named_index_.index_.nR,
        particle_system.named_index_.index_,
        criterion,
        action!,
        parameter,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
end

# * ==================== search ==================== * #

@inline function search!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    domain::AbstractDomain{IT, FT, Dimension},
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary};
    parameter = kDefaultParameter,
    launch::AbstractLaunch = kDefaultLaunch,
    action!::Function = defaultSelfaction!,
    criterion::Function = defaultCriterion,
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    Class.clean!(neighbour_system)
    n_particles::IT = Class.count(particle_system)
    host_insertParticlesIntoCells!(launch, particle_system, domain, neighbour_system, n_particles, n_threads)
    host_findNeighbourParticlesFromCells!(
        launch,
        particle_system,
        domain,
        neighbour_system,
        parameter,
        n_particles,
        criterion,
        action!,
        n_threads,
    )
    return nothing
end
