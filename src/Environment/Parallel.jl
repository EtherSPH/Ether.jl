#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 14:33:05
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== const CPU ==================== * #

const kCPUContainerType = Array
const kCPUBackend = KernelAbstractions.CPU()

# * ==================== AbstractParallel && Parallel ==================== * #

abstract type AbstractParallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} end

struct Parallel{IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend} <:
       AbstractParallel{IT, FT, CT, Backend} end

const ParallelCPU{IT, FT} = Parallel{IT, FT, kCPUContainerType, kCPUBackend} where {IT <: Integer, FT <: AbstractFloat}

function Base.show(
    io::IO,
    ::Parallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    println(io, "Parallel{$IT, $FT, $CT, $Backend}()")
end

# * ==================== get for AbstractParallel ==================== * #

@inline function get_inttype(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return IT
end

@inline function get_floattype(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return FT
end

@inline function get_containertype(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return CT
end

@inline function get_backend(
    ::AbstractParallel{IT, FT, CT, Backend},
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    return Backend
end

# * ==================== convert for AbstractParallel ==================== * #

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::IntType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, IntType <: Integer}
    return IT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::FloatType,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, FloatType <: AbstractFloat}
    return FT(x)
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:Integer}}
    if CT === AT
        if IT === eltype(x)
            return deepcopy(x)
        else
            return IT.(x)
        end
    else
        return CT(IT.(x))
    end
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::AT,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend, AT <: AbstractArray{<:AbstractFloat}}
    if CT === AT
        if FT === eltype(x)
            return deepcopy(x)
        else
            return FT.(x)
        end
    else
        return CT(FT.(x))
    end
end

@inline function (parallel::AbstractParallel{IT, FT, CT, Backend})(
    x::NamedTuple,
) where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    name_s = fieldnames(typeof(x))
    value_s = values(x)
    converted_value_s = map(value_s) do item
        if typeof(item) <: Real
            return parallel(item)
        else
            return item
        end
    end
    return NamedTuple{name_s}(converted_value_s)
end

@inline to(parallel, x) = parallel(x)
