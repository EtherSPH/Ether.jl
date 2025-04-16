#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/15 20:01:56
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "DataIO" begin
    config_dict = DataIO.template()
    replace!(config_dict, parallel)
    @test config_dict["parallel"]["backend"] == DEVICE
    domain = DataIO.Domain(config_dict)
    @test Class.get_gap(domain) ≈ config_dict["domain"]["gap"]
    @test Class.get_first_x(domain) ≈ config_dict["domain"]["first_x"]
    @test Class.get_first_y(domain) ≈ config_dict["domain"]["first_y"]
    @test Class.get_last_x(domain) ≈ config_dict["domain"]["last_x"]
    @test Class.get_last_y(domain) ≈ config_dict["domain"]["last_y"]
    n_particles = 100
    config_dict["particle_system"]["n_particles"] = n_particles
    config_dict["particle_system"]["capacity_expand"] = "n -> n * 2"
    particle_system = DataIO.ParticleSystem(config_dict, parallel)
    @test Class.get_n_particles(particle_system) == n_particles
    @test Class.get_n_capacity(particle_system) == n_particles * 2
    neighbour_system = DataIO.NeighbourSystem(config_dict, parallel)
    @test Class.get_n_cells(neighbour_system) == Class.get_n(domain)
    test_path = joinpath(@__DIR__, "test")
    config_dict["writer"]["path"] = test_path
    writer = DataIO.Writer(config_dict)
    @test DataIO.get_path(writer) == test_path
    DataIO.mkdir(writer)
    @test isdir(test_path)
    @test isdir(joinpath(test_path, "config"))
    @test isdir(joinpath(test_path, "raw"))
    replace!(config_dict, parallel)
    replace!(config_dict, domain)
    Class.set_n_particles!(particle_system, 10)
    replace!(config_dict, particle_system)
    replace!(config_dict, neighbour_system)
    @test config_dict["particle_system"]["n_particles"] == 10
    @test config_dict["particle_system"]["capacity_expand"] == "n -> n + 190"

    DataIO.save(config_dict; format = :all)
    @test isfile(joinpath(test_path, "config", "config.json"))
    @test isfile(joinpath(test_path, "config", "config.yaml"))
    cpu_particle_system = Class.mirror(particle_system)
    appendix = DataIO.appendix()
    for _ in 1:10
        DataIO.wait!(writer)
        DataIO.save!(writer, cpu_particle_system, appendix)
        appendix["WriteStep"] += 1
    end
    DataIO.wait!(writer)
    DataIO.readRaw!(writer)
    @test length(writer) == 10
    ps = DataIO.load(writer; appendix = appendix, step = :last, format = :yaml)
    ps = DataIO.load(writer; appendix = appendix, step = :last, format = :json)
    @test Class.count(ps) == 10

    DataIO.rmdir(writer)
    @test !isdir(test_path)
    @test !isdir(joinpath(test_path, "config"))
    @test !isdir(joinpath(test_path, "raw"))
end
