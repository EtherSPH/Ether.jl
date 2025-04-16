#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 18:13:22
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Mean

@inline @fastmath function arithmetic(a::Real, b::Real)
    return typeof(a)(0.5) * (a + b)
end

@inline @fastmath function invarithmetic(a::Real, b::Real)
    return 2 / (a + b)
end

@inline @fastmath function geometric(a::Real, b::Real)
    return sqrt(a * b)
end

@inline @fastmath function invgeometric(a::Real, b::Real)
    return 1 / sqrt(a * b)
end

@inline @fastmath function harmonic(a::Real, b::Real)
    return 2 * a * b / (a + b)
end

@inline @fastmath function invharmonic(a::Real, b::Real)
    return (a + b) / (2 * a * b)
end

@inline @fastmath function quadratic(a::Real, b::Real)
    return sqrt(a * a + b * b)
end

@inline @fastmath function invquadratic(a::Real, b::Real)
    return 1 / sqrt(a * a + b * b)
end

end # module Mean
