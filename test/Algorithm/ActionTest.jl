#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 16:36:06
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Algorithm" begin
    dim = 2
    n_particles = 10
    neighbour_count = 40
    n_neighbours = 4
    int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_named_tuple = (
        PositionVec = dim,
        Mass = 1,
        Density = 1,
        Volume = 1,
        VelocityVec = dim,
        dVelocityVec = dim,
        dDensity = 1,
        Pressure = 1,
        StrainMat = dim * dim,
        dStrainMat = dim * dim,
        StressMat = dim * dim,
        nRVec = dim * neighbour_count,
        nR = neighbour_count,
        nW = neighbour_count,
        nDW = neighbour_count,
    )
    ps = Class.ParticleSystem(Environment.Dimension2D, parallel, n_particles, int_named_tuple, float_named_tuple)
    cpu_ps = Class.mirror(ps)
    param = (k = FT(2), b = IT(2))
    cpu_ps.base_.int_[:, 3] .= n_neighbours
    Class.to!(ps, cpu_ps)
    @inline function self!(@self_args)
        @inbounds FLOAT[I, INDEX.PositionVec] = PARAMETER[1]
        @inbounds FLOAT[I, INDEX.PositionVec + 1] = PARAMETER[2]
    end
    @inline function inter!(@inter_args)
        @inbounds FLOAT[I, INDEX.PositionVec] *= PARAMETER.k
        @inbounds FLOAT[I, INDEX.PositionVec + 1] *= PARAMETER.k
        @inbounds FLOAT[I, INDEX.PositionVec] += PARAMETER.b
        @inbounds FLOAT[I, INDEX.PositionVec + 1] += PARAMETER.b
    end
    some_array = CT(FT[1.0, 2.0])
    Algorithm.selfaction!(ps, self!; parameter = some_array, launch = Algorithm.kDynamic)
    Algorithm.interaction!(ps, inter!; parameter = param, launch = Algorithm.kDynamic)
    Algorithm.selfaction!(ps, self!; parameter = some_array, launch = Algorithm.kStatic)
    Algorithm.interaction!(ps, inter!; parameter = param, launch = Algorithm.kStatic)
    Class.to!(cpu_ps, ps)
    function verify(x)
        for _ in 1:n_neighbours
            x = param.k * x + param.b
        end
        return x
    end
    @test cpu_ps.base_.float_[:, 1] ≈ [verify(1) for _ in 1:n_particles]
    @test cpu_ps.base_.float_[:, 2] ≈ [verify(2) for _ in 1:n_particles]
end
