#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 21:48:33
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "NeighbourSystemBase" begin
    domain = Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
    # 5 * 4 = 20 cells
    # 4 | 6 | 6 | 6 | 4
    # --|---|---|---|--
    # 6 | 9 | 9 | 9 | 6
    # --|---|---|---|--
    # 6 | 9 | 9 | 9 | 6
    # --|---|---|---|--
    # 4 | 6 | 6 | 6 | 4
    neighbour_system_base = Class.NeighbourSystemBase(parallel, domain)
    @test Array(neighbour_system_base.neighbour_cell_index_count_) ==
          [4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
    @test size(neighbour_system_base.neighbour_cell_index_count_) == (20,)
    @test size(neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
    cpu_neighbour_system_base = Class.mirror(neighbour_system_base)
    @test cpu_neighbour_system_base.neighbour_cell_index_count_ ==
          [4, 6, 6, 6, 4, 6, 9, 9, 9, 6, 6, 9, 9, 9, 6, 4, 6, 6, 6, 4]
    @test size(cpu_neighbour_system_base.neighbour_cell_index_count_) == (20,)
    @test size(cpu_neighbour_system_base.neighbour_cell_index_list_) == (20, 9)
end
