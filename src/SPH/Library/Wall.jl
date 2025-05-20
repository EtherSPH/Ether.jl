#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/18 14:33:13
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * SPH MODELING OF TSUNAMI WAVES, http://www.worldscientific.com/doi/abs/10.1142/9789812790910_0003, Rogers & Dalrymple - 2008
# * https://github.com/EtherSPH/EtherSPH.jl/blob/main/src/SPH/Library/BoundaryForce.jl
@inline function iCompulsive!(
    ::Type{DIMENSION},
    I::Integer,
    NI::Integer,
    INT,
    FLOAT,
    INDEX::NamedTuple,
    PARAMETER;
    c::Real = 0.0,
    h::Real = 1.0,
)::Nothing where {N, DIMENSION <: AbstractDimension{N}}
    psi = @float 0.0
    for i::@int() in 0:(N - 1)
        @inbounds psi += @rvec(@ij, i) * @nvec(@j, i)
    end
    psi = abs(psi)
    xi = sqrt(max(@float(0.0), @r(@ij) * @r(@ij) - psi * psi))
    eta = psi / @gap(@j)
    if eta > 1 || xi > @gap(@j)
        return nothing
    end
    p_xi = abs(1 + cos(pi * xi / @gap(@j))) * @float(0.5)
    verticle_velocity = @float(0.0)
    for i::@int() in 0:(N - 1)
        @inbounds verticle_velocity += (@u(@i, i) - @u(@j, i)) * @nvec(@j, i)
    end
    beta = @int(verticle_velocity > 0 ? 0 : 1)
    c0 = @float(c)
    h0 = @float(h)
    r_psi = (@float(0.01) * c0 * c0 + beta * c0 * abs(verticle_velocity)) * abs(1 - eta) / (sqrt(eta) * h0)
    coefficient = r_psi * p_xi
    for i::@int() in 0:(N - 1)
        @inbounds @du(@i, i) += coefficient * @nvec(@j, i)
    end
    return nothing
end
