#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/15 16:15:56
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline template(; dicttype = OrderedDict) = JSON.parsefile(joinpath(@__DIR__, "template.json"); dicttype = dicttype)

@inline function save(config_dict::AbstractDict; format = :all)::Nothing
    path = config_dict["writer"]["path"]
    path = joinpath(path, "config")
    Utility.assuredir(path)
    if format == :json
        open(joinpath(path, "config.json"), "w") do io
            JSON.print(io, config_dict, 4)
        end
        return nothing
    elseif format == :yaml
        YAML.write_file(joinpath(path, "config.yaml"), config_dict)
        return nothing
    elseif format == :all
        for form in [:json, :yaml]
            save(config_dict; format = form)
        end
    else
        error("Unsupported format: $format\nsupport: :json, :yaml, :all")
    end
    return nothing
end

# * ==================== Input ==================== * #

@inline function IT(config_dict::AbstractDict)
    return eval(Meta.parse(config_dict["parallel"]["int"]))
end

@inline function FT(config_dict::AbstractDict)
    return eval(Meta.parse(config_dict["parallel"]["float"]))
end

@inline function dimension(config_dict::AbstractDict)
    return config_dict["domain"]["dimension"]
end

@inline function Dimension(config_dict::AbstractDict)
    dim = DataIO.dimension(config_dict)
    if dim == 1
        return Environment.Dimension1D
    elseif dim == 2
        return Environment.Dimension2D
    elseif dim == 3
        return Environment.Dimension3D
    else
        error("Unsupported dimension: $dim")
    end
end

@inline function PeriodicBoundary(config_dict::AbstractDict)
    if config_dict["neighbour_system"]["periodic_boundary"]["type"] == "none"
        return Class.NonePeriodicBoundary
    elseif config_dict["neighbour_system"]["periodic_boundary"]["type"] == "2D"
        axis = config_dict["neighbour_system"]["periodic_boundary"]["axis"]
        return Class.PeriodicBoundary2D{axis[1], axis[2]}
    elseif config_dict["neighbour_system"]["periodic_boundary"]["type"] == "3D"
        axis = config_dict["neighbour_system"]["periodic_boundary"]["axis"]
        return Class.PeriodicBoundary3D{axis[1], axis[2], axis[3]}
    else
        error("Unsupported periodic boundary type: $(config_dict["neighbour_system"]["periodic_boundary"]["type"])")
    end
end

@inline function ParallelExpr(config_dict::AbstractDict)::Expr
    IT = config_dict["parallel"]["int"]
    FT = config_dict["parallel"]["float"]
    CT = Environment.kNameToContainer[config_dict["parallel"]["backend"]]
    Backend = Environment.kNameToBackend[config_dict["parallel"]["backend"]]
    Device = config_dict["parallel"]["device"]
    return Meta.parse(
        "const parallel = Environment.Parallel{$IT, $FT, $CT, $Backend}(); KernelAbstractions.device!($Backend, $Device)",
    )
end

@inline function Domain(config_dict::AbstractDict)
    IT = DataIO.IT(config_dict)
    FT = DataIO.FT(config_dict)
    if config_dict["domain"]["dimension"] == 2
        return Class.Domain2D{IT, FT}(
            config_dict["domain"]["gap"],
            config_dict["domain"]["first_x"],
            config_dict["domain"]["first_y"],
            config_dict["domain"]["last_x"],
            config_dict["domain"]["last_y"],
        )
    elseif config_dict["domain"]["dimension"] == 3
        # TODO: add 3D support
    else
        error("Unsupported dimension: $(config_dict["domain"]["dimension"])")
    end
end

@inline function ParticleSystem(
    config_dict::AbstractDict,
    parallel::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    n_particles = config_dict["particle_system"]["n_particles"]
    f = eval(Meta.parse(config_dict["particle_system"]["capacity_expand"]))
    n_capacity = Base.invokelatest(f, n_particles)
    int_named_tuple = Utility.dict2namedtuple(config_dict["particle_system"]["int_named_tuple"])
    float_named_tuple = Utility.dict2namedtuple(config_dict["particle_system"]["float_named_tuple"])
    Dimension = DataIO.Dimension(config_dict)
    return Class.ParticleSystem(Dimension, parallel, n_particles, n_capacity, int_named_tuple, float_named_tuple)
end

@inline function NeighbourSystem(
    config_dict::AbstractDict,
    parallel::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    domain = Domain(config_dict)
    N = config_dict["domain"]["dimension"]
    active_pair = [v[1] => v[2] for v in config_dict["neighbour_system"]["active_pair"]]
    periodic_boundary = DataIO.PeriodicBoundary(config_dict)
    return Class.NeighbourSystem(
        periodic_boundary,
        parallel,
        domain,
        active_pair;
        max_neighbour_number = config_dict["neighbour_system"]["max_neighbour_number"],
        n_threads = config_dict["neighbour_system"]["n_threads"],
    )
end

@inline function Writer(config_dict::AbstractDict)::AbstractWriter
    path = config_dict["writer"]["path"]
    file_name = config_dict["writer"]["raw_name"]
    connector = config_dict["writer"]["connector"]
    digits = config_dict["writer"]["digits"]
    suffix = config_dict["writer"]["suffix"]
    return Writer(path_ = path, raw_name_ = file_name, connector_ = connector, digits_ = digits, suffix_ = suffix)
end

# * ==================== Output ==================== * #

@inline function Base.replace!(
    config_dict::AbstractDict,
    ::AbstractParallel{IT, FT, CT, Backend},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    config_dict["parallel"]["int"] = "$IT"
    config_dict["parallel"]["float"] = "$FT"
    container_string = "$CT"
    name = Environment.kContainerToName[container_string]
    config_dict["parallel"]["backend"] = name
    device = KernelAbstractions.device(Backend)
    config_dict["parallel"]["device"] = device
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    domain::AbstractDomain{IT, FT, Dimension},
)::Nothing where {IT <: Integer, FT <: AbstractFloat, N, Dimension <: AbstractDimension{N}}
    config_dict["domain"]["dimension"] = N
    config_dict["domain"]["gap"] = Class.get_gap(domain)
    config_dict["domain"]["first_x"] = Class.get_first_x(domain)
    config_dict["domain"]["first_y"] = Class.get_first_y(domain)
    config_dict["domain"]["last_x"] = Class.get_last_x(domain)
    config_dict["domain"]["last_y"] = Class.get_last_y(domain)
    if N == 3
        config_dict["domain"]["first_z"] = Class.get_first_z(domain)
        config_dict["domain"]["last_z"] = Class.get_last_z(domain)
    end
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    particle_system::AbstractParticleSystem{IT, FT, CT, Backend, Dimension},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
}
    n_particles = Class.get_n_particles(particle_system)
    n_capacity = Class.get_n_capacity(particle_system)
    config_dict["particle_system"]["n_particles"] = n_particles
    config_dict["particle_system"]["capacity_expand"] = "n -> n + $(n_capacity - n_particles)"
    config_dict["particle_system"]["int_named_tuple"] =
        Utility.namedtuple2dict(particle_system.named_index_.int_capacity_)
    config_dict["particle_system"]["float_named_tuple"] =
        Utility.namedtuple2dict(particle_system.named_index_.float_capacity_)
    return nothing
end

@inline function Base.replace!(
    config_dict::AbstractDict,
    neighbour_system::AbstractNeighbourSystem{IT, FT, CT, Backend, Dimension, PeriodicBoundary},
)::Nothing where {
    IT <: Integer,
    FT <: AbstractFloat,
    CT <: AbstractArray,
    Backend,
    N,
    Dimension <: AbstractDimension{N},
    PeriodicBoundary <: AbstractPeriodicBoundary,
}
    active_pair = [[ac.first, ac.second] for ac in neighbour_system.active_pair_.pair_vector_]
    config_dict["neighbour_system"]["active_pair"] = active_pair
    config_dict["neighbour_system"]["max_neighbour_number"] =
        size(neighbour_system.base_.contained_particle_index_list_, 2)
    if PeriodicBoundary == Class.NonePeriodicBoundary
        config_dict["neighbour_system"]["periodic_boundary"]["type"] = "none"
    elseif PeriodicBoundary <: Class.PeriodicBoundary2D
        config_dict["neighbour_system"]["periodic_boundary"]["type"] = "2D"
        config_dict["neighbour_system"]["periodic_boundary"]["axis"][1] = PeriodicBoundary.parameters[1]
        config_dict["neighbour_system"]["periodic_boundary"]["axis"][2] = PeriodicBoundary.parameters[2]
    elseif PeriodicBoundary <: Class.PeriodicBoundary3D
        config_dict["neighbour_system"]["periodic_boundary"]["type"] = "3D"
        config_dict["neighbour_system"]["periodic_boundary"]["axis"][1] = PeriodicBoundary.parameters[1]
        config_dict["neighbour_system"]["periodic_boundary"]["axis"][2] = PeriodicBoundary.parameters[2]
        config_dict["neighbour_system"]["periodic_boundary"]["axis"][3] = PeriodicBoundary.parameters[3]
    end
    return nothing
end

@inline function Base.replace!(config_dict::AbstractDict, writer::AbstractWriter)::Nothing
    config_dict["writer"]["path"] = get_path(writer)
    config_dict["writer"]["raw_name"] = writer.raw_name_
    config_dict["writer"]["connector"] = writer.connector_
    config_dict["writer"]["digits"] = writer.digits_
    config_dict["writer"]["suffix"] = writer.suffix_
    return nothing
end
