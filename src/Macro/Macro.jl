#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 15:15:27
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

module Macro

macro self_args()
    return esc(:((DIMENSION, I, INT, FLOAT, INDEX, PARAMETER)...))
end

macro inter_args()
    return esc(:((DIMENSION, I, NI, INT, FLOAT, INDEX, PARAMETER)...))
end

macro criterion_args()
    return esc(:((DIMENSION, I, J, INT, FLOAT, INDEX, PARAMETER, domain, dr_square)...))
end

macro int()
    return esc(:(eltype(INT)))
end

macro int(x)
    return esc(:(eltype(INT)($x)))
end

macro float()
    return esc(:(eltype(FLOAT)))
end

macro float(x)
    return esc(:(eltype(FLOAT)($x)))
end

macro i()
    return esc(:(I))
end

macro j()
    return esc(:(INT[I, INDEX.nIndex + NI]))
end

macro ij()
    return esc(:(NI))
end

macro ci()
    return esc(:(I))
end

macro cj()
    return esc(:(J))
end

# an amazing macro allow use `interaction` in `selfaction` arg-passing
macro with_neighbours(expr)
    return esc(:(NI = eltype(INT)(0);
    @inbounds while NI < INT[I, INDEX.nCount]
        $expr
        NI += 1
    end))
end

export @self_args, @inter_args, @criterion_args
export @int, @float
export @i, @j, @ij
export @ci, @cj
export @with_neighbours

end # module Macro
