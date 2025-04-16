#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 18:17:59
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@generated function power(x, ::Val{N}) where {N}
    if N == 0
        return :(1)
    elseif N == 1
        return :(x)
    elseif N == 2
        return :(x * x)
    elseif N == 3
        return :(x * x * x)
    elseif N == 4
        return :(temp = x * x;
        temp * temp)
    elseif N == 5
        return :(temp = x * x;
        temp * temp * x)
    elseif N == 6
        return :(temp = x * x * x;
        temp * temp)
    elseif N == 7
        return :(temp = x * x * x;
        temp * temp * x)
    elseif N == 8
        return :(temp = x * x;
        temp = temp * temp;
        temp * temp)
    else
        expr = :(x)
        for _ in 2:N
            expr = :($expr * x)
        end
        return expr
    end
end
