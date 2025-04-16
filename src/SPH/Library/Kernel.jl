#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 22:23:16
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * ==================== Value ==================== * #

@inline function iValue(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    kernel::AbstractKernel;
)::@float() where {N, DIMENSION <: AbstractDimension{N}}
    return Kernel.value(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
end

@inline function iValue(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    h_inv::Real,
    kernel::AbstractKernel;
)::@float() where {N, DIMENSION <: AbstractDimension{N}}
    return Kernel._value(@r(@ij), hinv, kernel)
end

@inline function iValue!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    kernel::AbstractKernel;
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @w(@ij) = iValue(@inter_args, kernel)
    return nothing
end

@inline function iValue!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    h_inv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @w(@ij) = iValue(@inter_args, h_inv, kernel)
    return nothing
end

# * ==================== Gradient ==================== * #

@inline function iGradient(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    kernel::AbstractKernel;
)::@float() where {N, DIMENSION <: AbstractDimension{N}}
    return Kernel.gradient(@r(@ij), Math.Mean.arithmetic(@h(@i), @h(@j)), kernel)
end

@inline function iGradient(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    h_inv::Real,
    kernel::AbstractKernel;
)::@float() where {N, DIMENSION <: AbstractDimension{N}}
    return Kernel._gradient(@r(@ij), hinv, kernel)
end

@inline function iGradient!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    kernel::AbstractKernel;
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @w(@ij) = iGradient(@inter_args, kernel)
    return nothing
end

@inline function iGradient!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    h_inv::Real,
    kernel::AbstractKernel;
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    @inbounds @w(@ij) = iGradient(@inter_args, h_inv, kernel)
    return nothing
end

# * ==================== ValueGradient ==================== * #

@inline function iValueGradient!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER,
    kernel::AbstractKernel;
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    hinv::@float() = Math.Mean.invharmonic(@h(@i), @h(@j))
    @inbounds @hinv(@ij) = hinv
    @inbounds @w(@ij) = Kernel._value(@r(@ij), hinv, kernel)
    @inbounds @dw(@ij) = Kernel._gradient(@r(@ij), hinv, kernel)
    return nothing
end
