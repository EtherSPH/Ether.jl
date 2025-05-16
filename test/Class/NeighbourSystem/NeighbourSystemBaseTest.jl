#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:48:33
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "NeighbourSystemBase" begin
    @testset "NeighbourSystemBase 2D" begin
        domain = Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        neighbour_system_base = Class.NeighbourSystemBase(parallel, domain)
        @test size(neighbour_system_base.neighbour_cell_index_count_) == (20,)
        @test size(neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
        cpu_neighbour_system_base = Class.mirror(neighbour_system_base)
        @test size(cpu_neighbour_system_base.neighbour_cell_index_count_) == (20,)
        @test size(cpu_neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
    end
    @testset "NeighbourSystemBase 3D" begin
        domain = Class.Domain3D{IT, FT}(0.15, 0, 0, 0, 0.4, 0.5, 0.7)
        neighbour_system_base = Class.NeighbourSystemBase(parallel, domain)
        @test size(neighbour_system_base.neighbour_cell_index_count_) == (24,)
        @test size(neighbour_system_base.neighbour_cell_index_list_) == (24, 27)
        cpu_neighbour_system_base = Class.mirror(neighbour_system_base)
        @test size(cpu_neighbour_system_base.neighbour_cell_index_count_) == (24,)
        @test size(cpu_neighbour_system_base.neighbour_cell_index_list_) == (24, 27)
    end
end
