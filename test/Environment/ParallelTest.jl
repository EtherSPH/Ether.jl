#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:10:57
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Parallel" begin
    @test Environment.get_inttype(parallel) == IT
    @test Environment.get_floattype(parallel) == FT
    @test Environment.get_containertype(parallel) == CT
    @test Environment.get_backend(parallel) == Backend
    @test parallel(UInt8(1)) == IT(1)
    @test parallel(1.0) ≈ FT(1.0)
    @test parallel(Int64.(1:3)) |> Array == IT[1, 2, 3]
    @test parallel(Float64.(1:3)) |> Array == FT[1.0, 2.0, 3.0]
    nt = (a = 1, b = 2.0)
    @test parallel(nt) == (a = IT(1), b = FT(2.0))
end
