#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:39:26
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

struct Gaussian{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::Gaussian{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(3)
@inline sigma(::Gaussian{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / sqrt(pi))
@inline sigma(::Gaussian{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / pi)
@inline sigma(::Gaussian{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(1 / sqrt(pi^3))

@inline @fastmath function _value(
    r::Real,
    hinv::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(3.0)
        return sigma(kernel) * Math.power(hinv, Val(N)) * exp(-q * q)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    hinv::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(3.0)
        return -2 * sigma(kernel) * Math.power(hinv, Val(N + 1)) * q * exp(-q * q)
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::Gaussian{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
