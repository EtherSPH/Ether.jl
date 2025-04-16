#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/13 17:51:34
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractWriter end

# https://juliaio.github.io/JLD2.jl/dev/compression/
# - CodecZlib.ZlibCompressor: the default as it is widely used
# - CodecBzip2.Bzip2Compressor: can often times be faster
# - CodecLz4.Lz4Compressor: fast, but not compatible to the LZ4 shipped by HDF5
# - CodecZstd.ZstdCompressor: fast, wide range of compression size vs speed trade-offs
# * memo: when using python, `h5py` with `hdf5plugin` are required at the same time to read the compressed data.
# ! warning: compress may largely slow down the output speed, in case that frequent output step is not recommended
const kDefaultCompressor = CodecZstd.ZstdFrameCompressor()

@inline function appendix(; dicttype = OrderedDict)
    return dicttype("TMSTEP" => 0, "TimeValue" => 0.0, "TimeStamp" => "", "WriteStep" => 0)
end

# * ==================== AbsrtactWriter get function ==================== * #

@inline function get_path(writer::AbstractWriter)::AbstractString
    return writer.path_
end

@inline function get_config_path(writer::AbstractWriter)::AbstractString
    return joinpath(writer.path_, "config")
end

@inline function get_config_file(writer::AbstractWriter; format = :json)::AbstractString
    if format == :json
        return joinpath(get_config_path(writer), "config.json")
    elseif format == :yaml
        return joinpath(get_config_path(writer), "config.yaml")
    elseif format == :all
        return joinpath(get_config_path(writer), "config.json")
    else
        error("Unsupported format: $format\nsupport: :json, :yaml, :all")
    end
end

@inline function get_raw_path(writer::AbstractWriter)::AbstractString
    return joinpath(writer.path_, "raw")
end

@inline function get_raw_file_name(writer::AbstractWriter, step::Integer)::AbstractString
    return joinpath(
        get_raw_path(writer),
        string(writer.raw_name_, writer.connector_, string(step, pad = writer.digits_), writer.suffix_),
    )
end

@inline function get_raw_file_name(writer::AbstractWriter, step::Symbol)::AbstractString
    if step == :first
        return get_raw_file_name(writer, 0)
    elseif step == :last
        if isempty(writer.raw_file_list_)
            return get_raw_file_name(writer, 0)
        else
            return writer.raw_file_list_[end]
        end
    else
        error("Unsupported step: $step\nsupport: :first, :last")
    end
end

# * ==================== AbsrtactWriter dir function ==================== * #

@inline function mkdir(writer::AbstractWriter)::Nothing
    for get_path_function in [get_path, get_config_path, get_raw_path]
        Utility.assuredir(get_path_function(writer))
    end
    return nothing
end

@inline function rmdir(writer::AbstractWriter)::Nothing
    if isdir(get_path(writer))
        rm(get_path(writer), force = true, recursive = true)
    end
    return nothing
end

@inline function cleandir(writer::AbstractWriter)::Nothing
    rmdir(writer)
    mkdir(writer)
    return nothing
end

# * ==================== AbsrtactWriter save function ==================== * #

function Base.fetch(writer::AbstractWriter)::Nothing
    if isempty(writer.tasks_)
        return nothing
    else
        @inbounds Base.fetch(writer.tasks_[end])
    end
    return nothing
end

@inline function wait!(writer::AbstractWriter)::Nothing
    Base.fetch(writer)
    return nothing
end

@inline function Base.push!(writer::AbstractWriter, task::Base.Task)::Nothing
    push!(writer.tasks_, task)
    return nothing
end

@inline function save!(
    writer::AbstractWriter,
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension},
    appendix::AbstractDict;
    format = "yyyy_mm_dd_HH_MM_SS",
    compress = kDefaultCompressor,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    file_name = get_raw_file_name(writer, appendix["WriteStep"])
    file = JLD2.jldopen(file_name, "w"; compress = compress)
    appendix["TimeStamp"] = Utility.timestamp(; format = format)
    for (key, value) in appendix
        write(file, "appendix/$key", value)
    end
    task = Threads.@spawn begin
        @inbounds n_particles = Class.count(particle_system)
        @inbounds mask = particle_system.base_.is_alive_[1:n_particles] .== 1
        @inbounds write(file, "raw/int", particle_system.base_.int_[1:n_particles, :][mask, :])
        @inbounds write(file, "raw/float", particle_system.base_.float_[1:n_particles, :][mask, :])
        close(file)
    end
    push!(writer, task)
    return nothing
end

@inline function readRaw!(writer::AbstractWriter)::Nothing
    raw_path = get_raw_path(writer)
    if !isdir(raw_path)
        resize!(writer.raw_file_list_, 0)
        return nothing
    else
        raw_file_list = readdir(raw_path)
        resize!(writer.raw_file_list_, length(raw_file_list))
        for i in eachindex(writer.raw_file_list_)
            writer.raw_file_list_[i] = joinpath(raw_path, raw_file_list[i])
        end
    end
    return nothing
end

@inline function Base.length(writer::AbstractWriter)::Integer
    return length(writer.raw_file_list_)
end

@inline function Base.getindex(writer::AbstractWriter, index::Integer)::AbstractString
    if index > length(writer.raw_file_list_) || index < 1
        error("Index out of bounds")
    end
    @inbounds return writer.raw_file_list_[index]
end

@inline function load!(
    writer::AbstractWriter,
    particle_system::AbstractHostParticleSystem{IT, FT, Dimension};
    appendix::AbstractDict = appendix(),
    step = :first,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    file_name = get_raw_file_name(writer, step)
    JLD2.jldopen(file_name, "r") do file
        for key in keys(appendix)
            appendix[key] = file["appendix/$key"]
        end
        Class.set!(particle_system, file["raw/int"])
        Class.set!(particle_system, file["raw/float"])
    end
    Class.set_is_alive!(particle_system)
    return nothing
end

@inline function load(writer::AbstractWriter; appendix::AbstractDict = appendix(), step = :first, format = :json)
    if format == :json
        config_dict = JSON.parsefile(get_config_file(writer; format = :json); dicttype = OrderedDict)
    elseif format == :yaml
        config_dict = YAML.load_file(get_config_file(writer; format = :yaml); dicttype = OrderedDict)
    elseif format == :all
        config_dict = JSON.parsefile(get_config_file(writer; format = :json); dicttype = OrderedDict)
    else
        error("Unsupported format: $format\nsupport: :json, :yaml, :all")
    end
    IT = eval(Meta.parse(config_dict["parallel"]["int"]))
    FT = eval(Meta.parse(config_dict["parallel"]["float"]))
    parallel = Environment.ParallelCPU{IT, FT}()
    particle_system = DataIO.ParticleSystem(config_dict, parallel)
    load!(writer, particle_system; appendix = appendix, step = step)
    return particle_system
end

# * ==================== Writer function ==================== * #

@kwdef struct Writer <: AbstractWriter
    path_::AbstractString = "result/test"
    raw_name_::AbstractString = "result"
    connector_::AbstractString = "_"
    suffix_::AbstractString = ".jld2"
    digits_::Integer = 4
    raw_file_list_::Vector{String} = String[]
    tasks_::Vector{Base.Task} = Base.Task[]
end

function Base.show(io::IO, writer::Writer)
    println(io, "Writer{")
    println(io, "  path_ = ", writer.path_)
    println(io, "  config path: $(get_config_path(writer))")
    println(io, "  raw path: $(get_raw_path(writer))")
    println(io, "  raw_file_name: $(get_raw_file_name(writer, 0))")
    println(io, "  raw file number: $(length(writer.raw_file_list_))")
    println(io, "  output tasks: $(length(writer.tasks_))")
    println(io, "}")
end
