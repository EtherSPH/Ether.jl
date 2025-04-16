#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:40:06
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

struct WendlandC4{IT <: Integer, FT <: AbstractFloat, N} <: AbstractKernel{IT, FT, N} end

@inline radiusRatio(::WendlandC4{IT, FT, N}) where {IT <: Integer, FT <: AbstractFloat, N} = IT(2)
@inline sigma(::WendlandC4{IT, FT, 1}) where {IT <: Integer, FT <: AbstractFloat} = FT(5.0 / 8.0)
@inline sigma(::WendlandC4{IT, FT, 2}) where {IT <: Integer, FT <: AbstractFloat} = FT(9.0 / 4.0 / pi)
@inline sigma(::WendlandC4{IT, FT, 3}) where {IT <: Integer, FT <: AbstractFloat} = FT(495.0 / 256.0 / pi)

@inline @fastmath function _value(
    r::Real,
    hinv::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(2.0)
        return sigma(kernel) *
               Math.power(hinv, Val(N)) *
               Math.power(2 - q, Val(6)) *
               (q * (35 * q + 36) + 12) *
               FT(0.0013020833333333333)
    else
        return FT(0.0)
    end
end

@inline @fastmath function value(
    r::Real,
    h::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _value(r, 1 / h, kernel)
end

@inline @fastmath function _gradient(
    r::Real,
    hinv::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    q::FT = r * hinv
    if q < FT(2.0)
        return -sigma(kernel) *
               Math.power(hinv, Val(N + 1)) *
               q *
               Math.power(2 - q, Val(5)) *
               (2 + 5 * q) *
               FT(0.07291666666666667)
    else
        return FT(0.0)
    end
end

@inline @fastmath function gradient(
    r::Real,
    h::Real,
    kernel::WendlandC4{IT, FT, N},
)::FT where {IT <: Integer, FT <: AbstractFloat, N}
    return _gradient(r, 1 / h, kernel)
end
