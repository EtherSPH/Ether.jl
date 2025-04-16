#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 16:43:52
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "NamedIndex" begin
    dim = 2
    neighbour_count = 50
    int_capacity = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_capacity = (
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
    named_index = Class.NamedIndex{IT}(int_capacity, float_capacity)
    @test Class.get_n_int_capacity(named_index) == sum(collect(int_capacity))
    @test Class.get_n_float_capacity(named_index) == sum(collect(float_capacity))
    @test Class.get_index(named_index).Tag == 1
    @test Class.get_index(named_index).IsMovable == 2
    @test Class.get_index(named_index).nCount == 3
    @test Class.get_index(named_index).nIndex == 4
    @test Class.get_index(named_index).PositionVec == 1
    @test Class.get_index(named_index).Mass == dim + 1
    @test Class.get_index(named_index).Density == dim + 2
    @test Class.get_index(named_index).Volume == dim + 3
    @test Class.get_index(named_index).VelocityVec == dim + 4
    @test Class.get_index(named_index).dVelocityVec == dim + 4 + dim
    @test Class.get_index(named_index).dDensity == dim + 4 + dim + dim
    @test Class.get_index(named_index).Pressure == dim + 4 + dim + dim + 1
end
