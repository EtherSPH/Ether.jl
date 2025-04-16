#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:49:54
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@inline function iKernelFilter!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    w::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @wv(@i) += @float(w) * @vol(@j)
    @inbounds @wv_rho(@i) += @float(w) * @mass(@j)
    return nothing
end

@inline function sKernelFilter!(
    ::Type{DIMENSION},
    I::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    w0::Real = 0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @wv(@i) += @float(w0) * @vol(@i)
    @inbounds @wv_rho(@i) += @float(w0) * @mass(@i)
    @inbounds @rho(@i) = @wv_rho(@i) / @wv(@i)
    @wv(@i) = @float 0.0 # reset to zero
    @wv_rho(@i) = @float 0.0 # reset to zero
    return nothing
end
