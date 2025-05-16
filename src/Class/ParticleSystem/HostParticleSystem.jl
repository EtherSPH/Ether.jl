#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/15 21:56:54
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

const AbstractHostParticleSystem{IT, FT, CT, Dimension} =
    AbstractParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension}

const HostParticleSystem{IT, FT, Dimension} =
    ParticleSystem{IT, FT, Environment.kCPUContainerType, Environment.kCPUBackend, Dimension}

@inline function HostParticleSystem{IT, FT, Dimension}(
    n_particles::Integer,
    n_capacity::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    parallel = Environment.ParallelCPU{IT, FT}()
    return ParticleSystem(Dimension, parallel, n_particles, n_capacity, int_named_tuple, float_named_tuple)
end

@inline function HostParticleSystem{IT, FT, Dimension}(
    n_particles::Integer,
    int_named_tuple::NamedTuple,
    float_named_tuple::NamedTuple;
    capacityExpand::Function = defaultCapacityExpand,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return HostParticleSystem{IT, FT, Dimension}(
        n_particles,
        capacityExpand(n_particles),
        int_named_tuple,
        float_named_tuple,
    )
end

@inline function get_index(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::Integer where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return getfield(get_index(ps.named_index_), name)
end

@inline function get_capacity(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
)::Integer where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return getfield(get_capacity(ps.named_index_), name)
end

@inline function set_is_alive!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @inbounds particle_system.base_.is_alive_[1:count(particle_system)] .= 1
    return nothing
end

@inline function set!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    int::AbstractArray{<:Integer, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    m, n = size(int)
    @assert n == get_n_int_capacity(particle_system)
    @assert 0 < m <= get_n_capacity(particle_system)
    @inbounds particle_system.base_.int_[1:m, :] .= IT.(int)
    set_n_particles!(particle_system, m)
    return nothing
end

@inline function set!(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    float::AbstractArray{<:AbstractFloat, 2},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    m, n = size(float)
    @assert n == get_n_float_capacity(particle_system)
    @assert 0 < m <= get_n_capacity(particle_system)
    @inbounds particle_system.base_.float_[1:m, :] .= FT.(float)
    set_n_particles!(particle_system, m)
    return nothing
end

@inline function Base.reshape(
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    n_capacity::Integer,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = min(count(particle_system), n_capacity)
    parallel = Environment.ParallelCPU{IT, FT}()
    new_particle_system = ParticleSystem(
        Dimension,
        parallel,
        n_particles,
        n_capacity,
        particle_system.named_index_.int_capacity_,
        particle_system.named_index_.float_capacity_,
    )
    set_n_particles!(new_particle_system, n_particles)
    set!(new_particle_system, particle_system.base_.int_[1:n_particles, :])
    set!(new_particle_system, particle_system.base_.float_[1:n_particles, :])
    @inbounds new_particle_system.base_.is_alive_[1:n_particles] .= particle_system.base_.is_alive_[1:n_particles]
    @inbounds new_particle_system.base_.cell_index_[1:n_particles] .= particle_system.base_.cell_index_[1:n_particles]
    return new_particle_system
end

@inline function Base.merge(
    ps1::AbstractHostParticleSystem{IT, FT, Dimension},
    ps2::AbstractHostParticleSystem{IT, FT, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert get_n_int_capacity(ps1) == get_n_int_capacity(ps2) "int capacity mismatch"
    @assert get_n_float_capacity(ps1) == get_n_float_capacity(ps2) "float capacity mismatch"
    n_capacity = get_n_capacity(ps1) + get_n_capacity(ps2)
    new_ps = Base.reshape(ps1, n_capacity)
    n_particles_1 = count(ps1)
    n_particles_2 = count(ps2)
    start_index = n_particles_1 + 1
    end_index = n_particles_1 + n_particles_2
    @inbounds new_ps.base_.is_alive_[start_index:end_index] .= ps2.base_.is_alive_[1:n_particles_2]
    @inbounds new_ps.base_.cell_index_[start_index:end_index] .= ps2.base_.cell_index_[1:n_particles_2]
    @inbounds new_ps.base_.int_[start_index:end_index, :] .= ps2.base_.int_[1:n_particles_2, :]
    @inbounds new_ps.base_.float_[start_index:end_index, :] .= ps2.base_.float_[1:n_particles_2, :]
    set_n_particles!(new_ps, end_index)
    return new_ps
end

@inline Base.merge(ps...) = reduce(merge, ps)

@inline function Base.length(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
)::Integer where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    return count(ps)
end

@inline function Base.split(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    # * split by tag
    tags = @views ps[:Tag]
    first_tag, last_tag = extrema(tags)
    tag_number = last_tag - first_tag + 1
    ps_list = Vector{typeof(ps)}(undef, tag_number)
    for i in 1:tag_number
        mask = @views tags .== (first_tag + i - 1)
        ps_list[i] = ps[mask]
    end
    return ps_list
end

@inline function Base.split(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    tag::Integer,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    mask = @views ps[:Tag] .== tag
    return ps[mask]
end

@inline function Base.getindex(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    index::BitVector,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert count(ps) == length(index) "Invalid index size: $(length(index))"
    n_particles = sum(index)
    new_ps = ParticleSystem(
        Dimension,
        Environment.ParallelCPU{IT, FT}(),
        n_particles,
        n_particles,
        ps.named_index_.int_capacity_,
        ps.named_index_.float_capacity_,
    )
    set_n_particles!(new_ps, n_particles)
    set!(new_ps, ps.base_.int_[1:count(ps), :][index, :])
    set!(new_ps, ps.base_.float_[1:count(ps), :][index, :])
    @inbounds new_ps.base_.int_[1:n_particles, :] .= ps.base_.int_[1:count(ps), :][index, :]
    @inbounds new_ps.base_.float_[1:n_particles, :] .= ps.base_.float_[1:count(ps), :][index, :]
    return new_ps
end

@inline function Base.getindex(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = count(ps)
    if haskey(get_int_capacity(ps), name)
        index = get_index(ps, name)
        capacity = get_capacity(ps, name)
        if capacity == 1
            @inbounds return ps.base_.int_[1:n_particles, index]
        else
            @inbounds return ps.base_.int_[1:n_particles, index:(index + capacity - 1)]
        end
    elseif haskey(get_float_capacity(ps), name)
        index = get_index(ps, name)
        capacity = get_capacity(ps, name)
        if capacity == 1
            @inbounds return ps.base_.float_[1:n_particles, index]
        else
            @inbounds return ps.base_.float_[1:n_particles, index:(index + capacity - 1)]
        end
    else
        error("Invalid field name: $name")
    end
end

@inline function Base.getindex(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    name::Symbol,
    i::Integer,
) where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = count(ps)
    @assert 1 <= i <= n_particles "Index out of bounds: $i"
    if haskey(get_int_capacity(ps), name)
        index = get_index(ps, name)
        capacity = get_capacity(ps, name)
        n_particles = count(ps)
        if capacity == 1
            @inbounds return ps.base_.int_[i, index]
        else
            @inbounds return ps.base_.int_[i, index:(index + capacity - 1)]
        end
    elseif haskey(get_float_capacity(ps), name)
        index = get_index(ps, name)
        capacity = get_capacity(ps, name)
        n_particles = count(ps)
        if capacity == 1
            @inbounds return ps.base_.float_[i, index]
        else
            @inbounds return ps.base_.float_[i, index:(index + capacity - 1)]
        end
    else
        error("Invalid field name: $name")
    end
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::Integer,
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = count(ps)
    @assert haskey(get_int_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @inbounds ps.base_.int_[1:n_particles, index:(index + capacity - 1)] .= IT(value)
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractArray{<:Integer},
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_int_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @assert size(value, 1) <= get_n_capacity(ps) "Invalid value size: $(size(value))"
    @inbounds ps.base_.int_[1:size(value, 1), index:(index + capacity - 1)] .= IT.(value)
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::Integer,
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_int_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @inbounds ps.base_.int_[i, index:(index + capacity - 1)] .= IT(value)
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractArray{<:Integer},
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_int_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @assert length(value) <= capacity "Invalid value size: $(length(value))"
    @inbounds ps.base_.int_[i, index:(index + capacity - 1)] .= IT.(vec(value))
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractArray{<:AbstractFloat},
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_float_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @assert size(value, 1) <= get_n_capacity(ps) "Invalid value size: $(size(value))"
    @inbounds ps.base_.float_[1:size(value, 1), index:(index + capacity - 1)] .= FT.(value)
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractFloat,
    name::Symbol,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    n_particles = count(ps)
    @assert haskey(get_float_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @inbounds ps.base_.float_[1:n_particles, index:(index + capacity - 1)] .= FT(value)
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractArray{<:AbstractFloat},
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_float_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @assert length(value) <= capacity "Invalid value size: $(length(value))"
    @inbounds ps.base_.float_[i, index:(index + capacity - 1)] .= FT.(vec(value))
    return nothing
end

@inline function Base.setindex!(
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    value::AbstractFloat,
    name::Symbol,
    i::Integer,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    @assert haskey(get_float_capacity(ps), name) "Invalid field name: $name"
    index = get_index(ps, name)
    capacity = get_capacity(ps, name)
    @inbounds ps.base_.float_[i, index:(index + capacity - 1)] .= FT(value)
    return nothing
end
