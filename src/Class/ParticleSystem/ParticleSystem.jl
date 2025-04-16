#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:57:52
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

include("ParticleSystemBase.jl")
include("NamedIndex.jl")

# * ===================== ParticleSystem Definition ===================== * #

@inline function defaultCapacityExpand(n_particles::IT)::IT where {IT <: Integer}
    return n_particles
end

abstract type AbstractParticleSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
} end

struct ParticleSystem{
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    Dimension <: AbstractDimension,
} <: AbstractParticleSystem{IT, FT, CT, Backend, Dimension}
    n_particles_::Vector{IT}
    base_::ParticleSystemBase{IT, FT, CT, Backend}
    named_index_::NamedIndex{IT}
end

@inline function ParticleSystem(
    ::Type{Dimension},
    parallel::AbstractParallel{IT, FT, CT, Backend},
    n_particles::Integer,
    n_capacity::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
)::ParticleSystem{
    IT,
    FT,
    CT,
    Backend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_capacity::IT = parallel(n_capacity)
    n_particles = parallel(n_particles)
    named_index = NamedIndex{IT}(int_named_tuple, float_named_tuple)
    n_int_capacity = get_n_int_capacity(named_index)
    n_float_capacity = get_n_float_capacity(named_index)
    base = ParticleSystemBase(parallel, n_capacity, n_int_capacity, n_float_capacity)
    n_particles = [n_particles]
    is_alive = parallel(zeros(IT, n_capacity))
    @inbounds is_alive[1:n_particles[1]] .= 1
    Base.copyto!(base.n_particles_, n_particles)
    Base.copyto!(base.is_alive_, is_alive)
    return ParticleSystem{IT, FT, CT, Backend, Dimension}(n_particles, base, named_index)
end

@inline function ParticleSystem(
    ::Type{Dimension},
    parallel::AbstractParallel{IT, FT, CT, Backend},
    n_particles::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple;
    capacityExpand::Function = defaultCapacityExpand,
)::ParticleSystem{
    IT,
    FT,
    CT,
    Backend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    n_capacity = capacityExpand(n_particles)
    return ParticleSystem(Dimension, parallel, n_particles, n_capacity, int_named_tuple, float_named_tuple)
end

function Base.show(
    io::IO,
    particle_system::ParticleSystem{IT, FT, CT, Backend, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    println(io, "ParticleSystem{$IT, $FT, $Backend, $Dimension}(")
    println(io, "    n_particles: $(count(particle_system))")
    println(io, "    n_capacity: $(get_n_capacity(particle_system))")
    println(io, "    n_alive particles: $(get_n_alive_particles(particle_system))")
    println(io, "    n_int_capacity: $(get_n_int_capacity(particle_system))")
    println(io, "    n_float_capacity: $(get_n_float_capacity(particle_system))")
    println(io, "    index: $(get_index(particle_system))")
    println(io, ")")
end

# * ===================== ParticleSystem function ===================== * #

@inline function count!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    Base.copyto!(particle_system.n_particles_, particle_system.base_.n_particles_)
    return nothing
end

@inline function count!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend},
    n_particles::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    @inbounds particle_system.n_particles_[1] = IT(n_particles)
    Base.copyto!(particle_system.base_.n_particles_, particle_system.n_particles_)
    return nothing
end

@inline function count(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds return particle_system.n_particles_[1]
end

@inline function clean!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    set_n_particles!(particle_system, IT(0))
    return nothing
end

# * ===================== ParticleSystem set & get function ===================== * #

@inline function set_n_particles!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend},
    n_particles::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    @inbounds particle_system.n_particles_[1] = n_particles
    Base.copyto!(particle_system.base_.n_particles_, particle_system.n_particles_)
    return nothing
end

@inline function set_n_particles!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    Base.copyto!(particle_system.base_.n_particles_, particle_system.n_particles_)
    return nothing
end

@inline function get_n_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    @inbounds return particle_system.n_particles_[1]
end

@inline function get_n_alive_particles(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return sum(particle_system.base_.is_alive_)
end

@inline function get_n_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return length(particle_system.base_.is_alive_)
end

@inline function get_n_int_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.base_.int_, 2)
end

@inline function get_n_float_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::IT where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return size(particle_system.base_.float_, 2)
end

@inline function get_index(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.named_index_.index_
end

@inline function get_int_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.named_index_.int_capacity_
end

@inline function get_float_capacity(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::NamedTuple where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    return particle_system.named_index_.float_capacity_
end

# * ===================== ParticleSystem Data Transfer ===================== * #

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2, Dimension},
)::ParticleSystem{
    IT,
    FT,
    CT1,
    Backend1,
    Dimension,
} where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT1 <: AbstractArray,
    CT2 <: AbstractArray,
    Backend1,
    Backend2,
    Dimension <: AbstractDimension,
}
    n_particles = parallel(particle_system.n_particles_)
    base = mirror(parallel, particle_system.base_)
    named_index = deepcopy(particle_system.named_index_)
    return ParticleSystem{IT, FT, CT1, Backend1, Dimension}(n_particles, base, named_index)
end

@inline function mirror(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::ParticleSystem{
    IT,
    FT,
    Environment.kCPUContainerType,
    Environment.kCPUBackend,
    Dimension,
} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, Dimension <: AbstractDimension}
    parallel = Environment.ParallelCPU{IT, FT}()
    return mirror(parallel, particle_system)
end

@inbounds function serialto!(
    destination_particle_system::AbstractParticleSystem{IT, FT, CT1, Backend1},
    source_particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    @inbounds destination_particle_system.n_particles_[1] = source_particle_system.n_particles_[1]
    serialto!(destination_particle_system.base_, source_particle_system.base_)
    return nothing
end

@inbounds function asyncto!(
    destination_particle_system::AbstractParticleSystem{IT, FT, CT1, Backend1},
    source_particle_system::AbstractParticleSystem{IT, FT, CT2, Backend2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    @inbounds destination_particle_system.n_particles_[1] = source_particle_system.n_particles_[1]
    asyncto!(destination_particle_system.base_, source_particle_system.base_)
    return nothing
end

@inline to!(dst, src) = serialto!(dst, src) # default as serialto!

include("HostParticleSystem.jl")
