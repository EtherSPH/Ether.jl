#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:58:39
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== AbstractParticleSystemBase ==================== * #

abstract type AbstractParticleSystemBase{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} end

# * ==================== ParticleSystemBase ==================== * #

struct ParticleSystemBase{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} <:
       AbstractParticleSystemBase{IT, FT, CT, Backend}
    n_particles_::AbstractArray{IT, 1} # (1, ) on device, n_capacity_ >= n_particles_[1]
    is_alive_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0: dead, 1: alive
    cell_index_::AbstractArray{IT, 1} # (n_capacity, ) on device, 0: dead, 1: alive
    int_::AbstractArray{IT, 2} # (n_capacity, n_int_capacity) on device
    float_::AbstractArray{FT, 2} # (n_capacity, n_float_capacity) on device
end

@inline function ParticleSystemBase(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    n_capacity::Integer,
    n_int_capacity::Integer,
    n_float_capacity::Integer,
)::ParticleSystemBase{IT, FT, CT, Backend} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    n_capacity::IT = parallel(n_capacity)
    n_int_capacity::IT = parallel(n_int_capacity)
    n_float_capacity::IT = parallel(n_float_capacity)
    n_particles = parallel(zeros(IT, 1))
    is_alive = parallel(zeros(IT, n_capacity))
    cell_index = parallel(zeros(IT, n_capacity))
    int = parallel(zeros(IT, n_capacity, n_int_capacity))
    float = parallel(zeros(FT, n_capacity, n_float_capacity))
    return ParticleSystemBase{IT, FT, CT, Backend}(n_particles, is_alive, cell_index, int, float)
end

function Base.show(
    io::IO,
    particle_system_base::ParticleSystemBase{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    println(io, "ParticleSystemBase{$IT, $FT, $Backend}(")
    println(io, "    n_particles: $(Array(particle_system_base.n_particles_)[1])")
    println(io, "    n_capacity: $(length(particle_system_base.is_alive_))")
    println(io, "    n_alive particles: $(sum(particle_system_base.is_alive_))")
    println(io, "    n_int_capacity: $(size(particle_system_base.int_properties_, 2))")
    println(io, "    n_float_capacity: $(size(particle_system_base.float_properties_, 2))")
    println(io, ")")
end

# * ==================== AbstractParticleSystemBase ==================== * #

@inline function serialto!(
    destination::AbstractParticleSystemBase{IT, FT, CT1, Backend1},
    source::AbstractParticleSystemBase{IT, FT, CT2, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    Base.copyto!(destination.n_particles_, source.n_particles_)
    Base.copyto!(destination.is_alive_, source.is_alive_)
    Base.copyto!(destination.cell_index_, source.cell_index_)
    Base.copyto!(destination.int_, source.int_)
    Base.copyto!(destination.float_, source.float_)
    return nothing
end

@inline function asyncto!(
    destination::AbstractParticleSystemBase{IT, FT, CT1, Backend1},
    source::AbstractParticleSystemBase{IT, FT, CT2, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    task1 = Threads.@spawn begin
        Base.copyto!(destination.n_particles_, source.n_particles_)
        Base.copyto!(destination.is_alive_, source.is_alive_)
        Base.copyto!(destination.cell_index_, source.cell_index_)
        Base.copyto!(destination.int_, source.int_)
    end
    task2 = Threads.@spawn begin
        Base.copyto!(destination.float_, source.float_)
    end
    Base.fetch(task1)
    Base.fetch(task2)
    return nothing
end

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    particle_system_base::AbstractParticleSystemBase{IT, FT, CT2, Backend2},
) where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    n_capacity = length(particle_system_base.is_alive_)
    n_int_capacity = size(particle_system_base.int_, 2)
    n_float_capacity = size(particle_system_base.float_, 2)
    new_particle_system_base = ParticleSystemBase(parallel, n_capacity, n_int_capacity, n_float_capacity)
    serialto!(new_particle_system_base, particle_system_base)
    return new_particle_system_base
end

@inline function mirror(
    particle_system_base::AbstractParticleSystemBase{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    parallel = Environment.ParallelCPU{IT, FT}()
    return mirror(parallel, particle_system_base)
end
