#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 16:43:08
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "ParticleSystem" begin
    include("ParticleSystemBaseTest.jl")
    include("NamedIndexTest.jl")
    @testset "ParticleSystem" begin
        dim = 2
        neighbour_count = 50
        n_particles = 100
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
        capacityExpand(n)::typeof(n) = n + 100
        particle_system = Class.ParticleSystem(
            Environment.Dimension2D,
            parallel,
            n_particles,
            int_named_tuple,
            float_named_tuple;
            capacityExpand = capacityExpand,
        )
        Class.clean!(particle_system)
        @test Class.count(particle_system) == 0
        Class.set_n_particles!(particle_system, n_particles)
        @test Class.count(particle_system) == n_particles
        @test Class.get_n_particles(particle_system) == n_particles
        @test Class.get_n_alive_particles(particle_system) == n_particles
        @test Class.get_n_capacity(particle_system) == capacityExpand(n_particles)
        @test Class.get_n_int_capacity(particle_system) == sum(collect(int_named_tuple))
        @test Class.get_n_float_capacity(particle_system) == sum(collect(float_named_tuple))
        @test Class.get_index(particle_system) == Class.get_index(particle_system.named_index_)
        cpu_particle_system = Class.mirror(particle_system)
        @test Class.count(cpu_particle_system) == n_particles
        @test Class.get_n_particles(cpu_particle_system) == n_particles
        @test Class.get_n_alive_particles(cpu_particle_system) == n_particles
        @test Class.get_n_capacity(cpu_particle_system) == capacityExpand(n_particles)
        @test Class.get_n_int_capacity(cpu_particle_system) == sum(collect(int_named_tuple))
        @test Class.get_n_float_capacity(cpu_particle_system) == sum(collect(float_named_tuple))
        @test Class.get_index(cpu_particle_system) == Class.get_index(cpu_particle_system.named_index_)
        Class.to!(particle_system, cpu_particle_system)
        Class.asyncto!(particle_system, cpu_particle_system)
        @test Class.count(particle_system) == Class.count(cpu_particle_system)
        # * HostParticleSystem
        Class.set_n_particles!(cpu_particle_system, Class.get_n_capacity(cpu_particle_system))
        Class.set_is_alive!(cpu_particle_system)
        @test Class.get_n_alive_particles(cpu_particle_system) == Class.get_n_capacity(cpu_particle_system)
        Class.set!(cpu_particle_system, zeros(IT, n_particles, Class.get_n_int_capacity(cpu_particle_system)))
        Class.set!(cpu_particle_system, zeros(FT, n_particles, Class.get_n_float_capacity(cpu_particle_system)))
        @test sum(cpu_particle_system.base_.int_) == 0
        @test sum(cpu_particle_system.base_.float_) ≈ 0.0
        ps = reshape(cpu_particle_system, n_particles)
        ps = reshape(cpu_particle_system, capacityExpand(n_particles))
        @test Class.count(ps) == n_particles
        @test Class.get_n_capacity(ps) == capacityExpand(n_particles)
        n_repeat = 4
        ps = merge(ps, ps, ps, ps)
        @test Class.count(ps) == n_particles * n_repeat
        @test Class.get_n_capacity(ps) == capacityExpand(n_particles) * n_repeat
        ps[:Tag] = 1
        @test ps[:Tag] == ones(IT, n_particles * n_repeat)
        ps[:nIndex] = ones(IT, n_particles * n_repeat, neighbour_count)
        @test ps[:nIndex] == ones(IT, n_particles * n_repeat, neighbour_count)
        @test ps[:Tag, 1] == 1
        @test ps[:nIndex, 1] == ones(IT, neighbour_count)
        ps[:Mass] = FT(1)
        @test ps[:Mass] ≈ ones(FT, n_particles * n_repeat)
        ps[:PositionVec] = ones(FT, n_particles * n_repeat, dim)
        @test ps[:PositionVec] ≈ ones(FT, n_particles * n_repeat, dim)
        @test ps[:Mass, 1] ≈ FT(1.0)
        @test ps[:PositionVec, 1] ≈ ones(FT, dim)
    end
end
