#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:56:59
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractActivePair{IT <: Integer, CT <: AbstractArray, Backend} end

struct ActivePair{IT <: Integer, CT <: AbstractArray, Backend} <: AbstractActivePair{IT, CT, Backend}
    pair_vector_::Vector{Pair{IT, IT}}
    adjacency_matrix_::AbstractArray{IT, 2}
end

@inline function ActivePair(
    parallel::AbstractParallel{IT, FT, CT, Backend},
    pair_vector::AbstractVector{<:Pair{<:Integer, <:Integer}},
)::ActivePair{IT, CT, Backend} where {IT <: Integer, FT <: AbstractFloat, CT <: AbstractArray, Backend}
    active_pair_pair_vector = Vector{Pair{IT, IT}}()
    maximum_tag = IT(1)
    for pair in pair_vector
        push!(active_pair_pair_vector, Pair{IT, IT}(IT(pair.first), IT(pair.second)))
        maximum_tag = max(maximum_tag, pair.first, pair.second)
    end
    active_pair_adjacency_matrix = zeros(IT, maximum_tag, maximum_tag)
    for pair in active_pair_pair_vector
        active_pair_adjacency_matrix[pair.first, pair.second] = IT(1)
    end
    active_pair_adjacency_matrix = parallel(active_pair_adjacency_matrix)
    return ActivePair{IT, CT, Backend}(active_pair_pair_vector, active_pair_adjacency_matrix)
end

function Base.show(io::IO, active_pair::ActivePair{IT, CT, Backend}) where {IT <: Integer, CT <: AbstractArray, Backend}
    println(io, "ActivePair{", IT, ", ", CT, ", ", Backend, "}(")
    println(io, "pair_vector_: $(active_pair.pair_vector_)")
    println(io, "adjacency_matrix_: $(active_pair.adjacency_matrix_)")
    println(io, ")")
end

# * ===================== AbstractActivePair transfer ========================== * #

@inline function serialto!(
    destination_active_pair::AbstractActivePair{IT, CT1, Backend1},
    source_active_pair::AbstractActivePair{IT, CT2, Backend2};
)::Nothing where {IT <: Integer, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    resize!(destination_active_pair.pair_vector_, length(source_active_pair.pair_vector_))
    destination_active_pair.pair_vector_ .= source_active_pair.pair_vector_
    Base.copyto!(destination_active_pair.adjacency_matrix_, source_active_pair.adjacency_matrix_)
    return nothing
end

@inline function asyncto!(
    destination_active_pair::AbstractActivePair{IT, CT1, Backend1},
    source_active_pair::AbstractActivePair{IT, CT2, Backend2};
)::Nothing where {IT <: Integer, CT1 <: AbstractArray, CT2 <: AbstractArray, Backend1, Backend2}
    task = Threads.@spawn Base.copyto!(destination_active_pair.adjacency_matrix_, source_active_pair.adjacency_matrix_)
    resize!(destination_active_pair.pair_vector_, length(source_active_pair.pair_vector_))
    destination_active_pair.pair_vector_ .= source_active_pair.pair_vector_
    Base.fetch(task)
    return nothing
end

@inline function mirror(
    parallel::AbstractParallel{IT, FT, CT1, Backend1},
    active_pair::AbstractActivePair{IT, CT2, Backend2},
) where {IT <: Integer, FT <: AbstractFloat, CT1 <: AbstractArray, Backend1, CT2 <: AbstractArray, Backend2}
    new_active_pair = ActivePair(parallel, active_pair.pair_vector_)
    serialto!(new_active_pair, active_pair)
    return new_active_pair
end

@inline function mirror(
    active_pair::AbstractActivePair{IT, CT, Backend},
) where {IT <: Integer, CT <: AbstractArray, Backend}
    parallel = Environment.ParallelCPU{IT, Float32}()
    return mirror(parallel, active_pair)
end
