#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/14 14:53:06
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Utility

using Dates
using OrderedCollections

@inline function assuredir(path::AbstractString)::Nothing
    if !isdir(path)
        mkpath(path)
    end
    return nothing
end

@inline function timestamp(; format = "yyyy_mm_dd_HH_MM_SS")::String
    return Dates.format(now(), format)
end

@inline function namedtuple2dict(nt::NamedTuple; keytype = String, dicttype = OrderedDict)::dicttype
    key_s = map(k -> keytype(k), collect(keys(nt)))
    value_s = collect(values(nt))
    dict = dicttype(key_s .=> value_s)
    return dict
end

@inline function dict2namedtuple(dict::AbstractDict)::NamedTuple
    key_s = map(k -> Symbol(k), collect(keys(dict)))
    value_s = collect(values(dict))
    nt = NamedTuple{Tuple(key_s)}(value_s)
    return nt
end

end # module Utility
