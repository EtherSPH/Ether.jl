#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 16:43:39
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "ParticleSystemBase" begin
    n_capacity = 100
    int_n_capacity = 20
    float_n_capacity = 30
    particle_system_base = Class.ParticleSystemBase(parallel, n_capacity, int_n_capacity, float_n_capacity)
    @test size(particle_system_base.n_particles_) == (1,)
    @test size(particle_system_base.is_alive_) == (n_capacity,)
    @test size(particle_system_base.cell_index_) == (n_capacity,)
    @test size(particle_system_base.int_) == (n_capacity, int_n_capacity)
    @test size(particle_system_base.float_) == (n_capacity, float_n_capacity)
    cpu_particle_system_base = Class.mirror(particle_system_base)
    @test size(cpu_particle_system_base.n_particles_) == (1,)
    @test size(cpu_particle_system_base.is_alive_) == (n_capacity,)
    @test size(cpu_particle_system_base.cell_index_) == (n_capacity,)
    @test size(cpu_particle_system_base.int_) == (n_capacity, int_n_capacity)
    @test size(cpu_particle_system_base.float_) == (n_capacity, float_n_capacity)
end
