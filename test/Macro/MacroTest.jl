#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 21:51:27
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Macro" begin
    using Ether.Macro
    INDEX = (nIndex = 1,)
    INT = IT[
        3 1 4 1 5 9
        2 6 5 3 5 8
        9 7 9 3 2 3
        8 4 6 2 6 4
        3 3 8 3 2 7
        9 5 0 2 8 1
    ]
    FLOAT = FT[
        3.0 1.0 4.0
        1.0 5.0 9.0
        2.0 6.0 5.0
        3.0 5.0 8.0
        9.0 7.0 9.0
        3.0 2.0 3.0
    ]
    @test @int() == IT
    @test @float() == FT
    m, n = size(INT)
    for DIMENSION in [Environment.Dimension2D, Environment.Dimension3D]
        for I in 1:m
            for NI in 0:(n - 1)
                J = INT[I, INDEX.nIndex + NI]
                @test @i() == I
                @test @j() == J
                @test @ij() == NI
                @test @ci() == I
                @test @cj() == J
            end
        end
    end
end
