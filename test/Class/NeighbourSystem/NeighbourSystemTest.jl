#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:48:13
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "NeighbourSystem" begin
    include("NeighbourSystemBaseTest.jl")
    include("ActivePairTest.jl")
    @testset "NeighbourSystem 2D" begin
        domain = Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        active_pair = [1 => 1, 1 => 2, 2 => 1]
        periodic_boundary = Class.NonePeriodicBoundary
        neighbour_system = Class.NeighbourSystem(periodic_boundary, parallel, domain, active_pair)
        Class.clean!(neighbour_system)
        # * base
        @test Array(neighbour_system.base_.contained_particle_index_count_) ==
              zeros(IT, Class.get_n_cells(neighbour_system))
        @test Array(neighbour_system.base_.neighbour_cell_index_count_) ==
              IT[4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
        @test size(neighbour_system.base_.neighbour_cell_index_count_) == (20,)
        @test size(neighbour_system.base_.neighbour_cell_index_list_) == (20, 9)
        # * active pair
        @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
        @test Array(neighbour_system.active_pair_.adjacency_matrix_) == IT[
            1 1
            1 0
        ]
        cpu_neighbour_system = Class.mirror(neighbour_system)
        Class.clean!(cpu_neighbour_system)
        # * base
        @test Array(neighbour_system.base_.contained_particle_index_count_) ==
              cpu_neighbour_system.base_.contained_particle_index_count_
        @test cpu_neighbour_system.base_.neighbour_cell_index_count_ ==
              IT[4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
        @test size(cpu_neighbour_system.base_.neighbour_cell_index_count_) == (20,)
        @test size(cpu_neighbour_system.base_.neighbour_cell_index_list_) == (20, 9)
        # * active pair
        @test cpu_neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
        @test cpu_neighbour_system.active_pair_.adjacency_matrix_ == IT[
            1 1
            1 0
        ]
    end
    @testset "NeighbourSystem 3D" begin
        domain = Class.Domain3D{IT, FT}(0.15, 0, 0, 0, 0.4, 0.5, 0.7)
        active_pair = [1 => 1, 1 => 2, 2 => 1]
        periodic_boundary = Class.PeriodicBoundary3D{false, false, false}
        neighbour_system = Class.NeighbourSystem(periodic_boundary, parallel, domain, active_pair)
        Class.clean!(neighbour_system)
        # * base
        @test Array(neighbour_system.base_.contained_particle_index_count_) ==
              zeros(IT, Class.get_n_cells(neighbour_system))
        @test Array(neighbour_system.base_.neighbour_cell_index_count_) ==
              IT[8, 8, 12, 12, 8, 8, 12, 12, 18, 18, 12, 12, 12, 12, 18, 18, 12, 12, 8, 8, 12, 12, 8, 8]
        @test size(neighbour_system.base_.neighbour_cell_index_count_) == (24,)
        @test size(neighbour_system.base_.neighbour_cell_index_list_) == (24, 27)
        # * active pair
        @test neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
        @test Array(neighbour_system.active_pair_.adjacency_matrix_) == IT[
            1 1
            1 0
        ]
        cpu_neighbour_system = Class.mirror(neighbour_system)
        Class.clean!(cpu_neighbour_system)
        # * base
        @test Array(neighbour_system.base_.contained_particle_index_count_) ==
              cpu_neighbour_system.base_.contained_particle_index_count_
        @test cpu_neighbour_system.base_.neighbour_cell_index_count_ ==
              IT[8, 8, 12, 12, 8, 8, 12, 12, 18, 18, 12, 12, 12, 12, 18, 18, 12, 12, 8, 8, 12, 12, 8, 8]
        @test size(cpu_neighbour_system.base_.neighbour_cell_index_count_) == (24,)
        @test size(cpu_neighbour_system.base_.neighbour_cell_index_list_) == (24, 27)
        # * active pair
        @test cpu_neighbour_system.active_pair_.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1)]
        @test cpu_neighbour_system.active_pair_.adjacency_matrix_ == IT[
            1 1
            1 0
        ]
    end
end
