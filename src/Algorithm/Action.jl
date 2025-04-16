#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 15:06:16
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== selfaction! ==================== * #

@inline function defaultSelfaction!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    return nothing
end

@kernel function device_selfaction!(
    DIMENSION::Type{Dimension},
    @Const(ps_is_alive),
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    action!::Function,
) where {N, Dimension <: AbstractDimension{N}}
    I::eltype(INT) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        action!(DIMENSION, I, INT, FLOAT, INDEX, PARAMETER)
    end
end

@inline function host_selfaction!(
    ::Static,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    parameter,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    n_particles = Class.count(particle_system)
    kernel_selfaction! = device_selfaction!(Backend, n_threads, (Int64(n_particles),))
    kernel_selfaction!(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        particle_system.named_index_.index_,
        parameter,
        action!,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function host_selfaction!(
    ::Dynamic,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    parameter,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    device_selfaction!(Backend, n_threads)(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        particle_system.named_index_.index_,
        parameter,
        action!,
        ndrange = (Class.count(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function selfaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    parameter = kDefaultParameter,
    launch::AbstractLaunch = kDefaultLaunch,
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    host_selfaction!(launch, particle_system, parameter, action!, n_threads)
    return nothing
end

# * ==================== interaction! ==================== * #

@inline function defaultInteraction!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    return nothing
end

@kernel function device_interaction!(
    DIMENSION::Type{Dimension},
    @Const(ps_is_alive),
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    index_nCount::IT,
    action!::Function,
) where {IT <: Integer, N, Dimension <: AbstractDimension{N}}
    I::eltype(INT) = @index(Global)
    @inbounds if ps_is_alive[I] == 1
        NI::IT = IT(0)
        @inbounds while NI < INT[I, index_nCount]
            action!(DIMENSION, I, NI, INT, FLOAT, INDEX, PARAMETER)
            NI += IT(1)
        end
    end
end

@inline function host_interaction!(
    ::Static,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    parameter,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    n_particles = Class.count(particle_system)
    kernel_interaction! = device_interaction!(Backend, n_threads, (Int64(n_particles),))
    kernel_interaction!(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        particle_system.named_index_.index_,
        parameter,
        particle_system.named_index_.index_.nCount,
        action!,
        ndrange = (n_particles,),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function host_interaction!(
    ::Dynamic,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    parameter,
    action!::Function,
    n_threads::Integer,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    device_interaction!(Backend, n_threads)(
        Dimension,
        particle_system.base_.is_alive_,
        particle_system.base_.int_,
        particle_system.base_.float_,
        particle_system.named_index_.index_,
        parameter,
        particle_system.named_index_.index_.nCount,
        action!,
        ndrange = (Class.count(particle_system),),
    )
    KernelAbstractions.synchronize(Backend)
    return nothing
end

@inline function interaction!(
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
    action!::Function;
    parameter = kDefaultParameter,
    launch::AbstractLaunch = kDefaultLaunch,
    n_threads::Integer = kDefaultThreadNumber,
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    host_interaction!(launch, particle_system, parameter, action!, n_threads)
    return nothing
end
