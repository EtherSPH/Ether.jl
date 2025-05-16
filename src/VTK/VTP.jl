#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/16 15:17:53
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module VTP

using WriteVTK
using ProgressMeter

using Ether.Environment
using Ether.Class
using Ether.DataIO

@inline function __generate(
    writer::AbstractWriter,
    ps::AbstractHostParticleSystem{IT, FT, Dimension},
    appendix::AbstractDict,
    name::AbstractString,
    data::AbstractVector{Symbol};
    with_step = true,
    compress = true,
    append = false,
    ascii = false,
)::Nothing where {IT <: Integer, FT <: AbstractFloat, Dimension <: AbstractDimension}
    if with_step == true
        file_name = name * string(appendix["WriteStep"], pad = writer.digits_) * ".vtp"
    else
        file_name = name * ".vtp"
    end
    file_name = joinpath(DataIO.get_vtp_path(writer), file_name)
    n = Class.count(ps)
    points = @views ps[:PositionVec]'
    cells = [MeshCell(PolyData.Verts(), [i]) for i in 1:n]
    vtk_grid(file_name, points, cells; compress = compress, append = append, ascii = ascii) do vtp
        if with_step == true
            for (key, value) in appendix
                vtp[key] = value
            end
        end
        for field in data
            capacity = Class.get_capacity(ps, field)
            field_str = string(field)
            if capacity == 1
                vtp[field_str] = @views ps[field]
            else
                vtp[field_str] = @views ps[field]'
            end
        end
    end
    return nothing
end

@inline function _generate(
    writer::AbstractWriter,
    i_file::Integer,
    tag_list::AbstractVector{<:Integer},
    with_step_tag_list::AbstractVector{<:Integer},
    names::AbstractVector{<:AbstractString},
    data::AbstractVector{Symbol};
    args...,
)::Nothing
    appendix = DataIO.appendix()
    particle_system = DataIO.load(writer; appendix = appendix, step = i_file)
    ps_list = [split(particle_system, tag) for tag in with_step_tag_list]
    for (i, ps) in enumerate(ps_list)
        @inbounds tag = with_step_tag_list[i]
        tag_index = findfirst(x -> x == tag, tag_list)
        @inbounds name = names[tag_index]
        __generate(writer, ps, appendix, name, data; with_step = true, args...)
    end
    return nothing
end

@inline function generate(
    writer::AbstractWriter,
    data::AbstractVector{Symbol};
    names = String[],
    n_threads = 1,
    args...,
)::Nothing
    @assert n_threads >= 1 && n_threads <= Threads.nthreads()
    DataIO.readRaw!(writer)
    n_files = length(writer)
    demo_step = 0
    demo_appendix = DataIO.appendix()
    demo_ps = DataIO.load(writer; appendix = demo_appendix, step = demo_step)
    demo_ps_list = split(demo_ps)
    tag_list = Int[]
    with_step_tag_list = Int[]
    if length(demo_ps_list) < length(names)
        names = names[1:length(demo_ps_list)]
    elseif length(demo_ps_list) > length(names)
        names = vcat(names, ["$(i)tag" for i in (length(names) + 1):length(demo_ps_list)])
    else
        nothing
    end
    for (i, ps) in enumerate(demo_ps_list)
        tag = ps[:Tag][1]
        push!(tag_list, tag)
        if ps[:IsMovable][1] == 1
            push!(with_step_tag_list, tag)
        else
            __generate(writer, ps, demo_appendix, names[i], data; with_step = false, args...)
        end
    end
    task_list = Vector{Task}(undef, n_threads)
    progress_bar = Progress(
        n_files,
        "Generating VTP files";
        dt = 0.5,
        color = :green,
        barglyphs = BarGlyphs("[=> ]"),
        showspeed = true,
    )
    for i_thread in 1:n_threads
        task_list[i_thread] = Threads.@spawn begin
            i_file = 0 + i_thread - 1
            while i_file < n_files
                _generate(writer, i_file, tag_list, with_step_tag_list, names, data; args...)
                i_file += n_threads
                next!(progress_bar)
            end
        end
    end
    for i_thread in 1:n_threads
        Base.fetch(task_list[n_threads + 1 - i_thread])
    end
    return nothing
end

end # module VTP
