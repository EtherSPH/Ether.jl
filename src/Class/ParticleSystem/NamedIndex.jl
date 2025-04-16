#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 16:16:32
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function Base.convert(IT::Type{<:Integer}, nt::NamedTuple)::NamedTuple
    key_set = keys(nt)
    value_set = collect(nt)
    value_set = map(x -> IT(x), value_set)
    return NamedTuple{key_set}(value_set)
end

@inline function accumulate(nt::NamedTuple)::NamedTuple
    key_set = keys(nt)
    value_set = collect(nt)
    new_value_set = similar(value_set)
    @inbounds new_value_set[1] = typeof(value_set[1])(1)
    n = length(value_set)
    if n <= 1
        return NamedTuple{key_set}(new_value_set)
    else
        for i in 2:n
            @inbounds new_value_set[i] = value_set[i - 1] + new_value_set[i - 1]
        end
    end
    return NamedTuple{key_set}(new_value_set)
end

# * ==================== AbstractNamedIndex ==================== * #

abstract type AbstractNamedIndex{IT} end

# * ==================== NamedIndex ==================== * #

struct NamedIndex{IT} <: AbstractNamedIndex{IT}
    int_capacity_::NamedTuple
    int_index_::NamedTuple
    n_int_capacity_::IT
    float_capacity_::NamedTuple
    float_index_::NamedTuple
    n_float_capacity_::IT
    index_::NamedTuple # a union of int_index_ and float_index_
    capacity_::NamedTuple # a union of int_capacity_ and float_capacity_
end

@inline function NamedIndex{IT}(
    int_capacity::NamedTuple,
    float_capacity::NamedTuple,
)::NamedIndex{IT} where {IT <: Integer}
    int_capacity = convert(IT, int_capacity)
    float_capacity = convert(IT, float_capacity)
    int_index = accumulate(int_capacity)
    float_index = accumulate(float_capacity)
    n_int_capacity = sum(collect(int_capacity))
    n_float_capacity = sum(collect(float_capacity))
    index = merge(int_index, float_index)
    capacity = merge(int_capacity, float_capacity)
    return NamedIndex{IT}(
        int_capacity,
        int_index,
        n_int_capacity,
        float_capacity,
        float_index,
        n_float_capacity,
        index,
        capacity,
    )
end

function Base.show(io::IO, named_index::NamedIndex{IT})::Nothing where {IT <: Integer}
    println(io, "NamedIndex{$IT}(")
    println(io, "    n_int_capacity_: $(named_index.n_int_capacity_)")
    println(io, "    n_float_capacity_: $(named_index.n_float_capacity_)")
    println(io, "    int_capacity_: $(named_index.int_capacity_)")
    println(io, "    float_capacity_: $(named_index.float_capacity_)")
    println(io, "    index_: $(named_index.index_)")
    println(io, ")")
end

# * ==================== get for NamedIndex ==================== * #

@inline function get_n_int_capacity(named_index::AbstractNamedIndex{IT})::IT where {IT <: Integer}
    return named_index.n_int_capacity_
end

@inline function get_n_float_capacity(named_index::AbstractNamedIndex{IT})::IT where {IT <: Integer}
    return named_index.n_float_capacity_
end

@inline function get_index(named_index::AbstractNamedIndex{IT})::NamedTuple where {IT <: Integer}
    return named_index.index_
end

@inline function get_capacity(named_index::AbstractNamedIndex{IT})::NamedTuple where {IT <: Integer}
    return named_index.capacity_
end

# * ==================== NamedIndex data transfer ==================== * #

@inline function mirror(named_index::NamedIndex{IT})::NamedIndex{IT} where {IT <: Integer}
    return deepcopy(named_index)
end

@inline function mirror(::AbstractParallel, named_index::NamedIndex{IT})::NamedIndex{IT} where {IT <: Integer}
    return mirror(named_index)
end
