#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 22:02:44
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "ActivePair" begin
    pair_vector = [1 => 1, 1 => 2, 2 => 1, 1 => 3, 3 => 1]
    active_pair = Class.ActivePair(parallel, pair_vector)
    @test active_pair.pair_vector_ == [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
    @test Array(active_pair.adjacency_matrix_) == IT[
        1 1 1
        1 0 0
        1 0 0
    ]
    cpu_active_pair = Class.mirror(active_pair)
    @test cpu_active_pair.pair_vector_ ==
          [IT(1) => IT(1), IT(1) => IT(2), IT(2) => IT(1), IT(1) => IT(3), IT(3) => IT(1)]
    @test cpu_active_pair.adjacency_matrix_ == IT[
        1 1 1
        1 0 0
        1 0 0
    ]
end
