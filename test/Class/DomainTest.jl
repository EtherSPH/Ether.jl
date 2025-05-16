#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/10 15:46:42
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Domain" begin
    @testset "Domain2D" begin
        x_0 = -1.0
        y_0 = -2.0
        x_1 = 3.0
        y_1 = 3.0
        gap = 0.15
        domain = Class.Domain2D{IT, FT}(gap, x_0, y_0, x_1, y_1)
        @test Class.dimension(domain) == 2
        @test Class.get_gap(domain) ≈ gap
        @test Class.get_gap_square(domain) ≈ gap * gap
        @test Class.get_n_x(domain) == 26
        @test Class.get_n_y(domain) == 33
        @test Class.get_n(domain) == 26 * 33
        @test Class.get_first_x(domain) ≈ x_0
        @test Class.get_first_y(domain) ≈ y_0
        @test Class.get_last_x(domain) ≈ x_1
        @test Class.get_last_y(domain) ≈ y_1
        @test Class.get_span_x(domain) ≈ x_1 - x_0
        @test Class.get_span_y(domain) ≈ y_1 - y_0
        @test Class.get_gap_x(domain) ≈ (x_1 - x_0) / 26
        @test Class.get_gap_y(domain) ≈ (y_1 - y_0) / 33
        @test Class.get_gap_x_inv(domain) ≈ 26 / (x_1 - x_0)
        @test Class.get_gap_y_inv(domain) ≈ 33 / (y_1 - y_0)

        @test Class.indexCartesianToLinear(IT(2), IT(3), domain) == 2 + (3 - 1) * 26
        @test Class.indexLinearToCartesian(IT(2 + (3 - 1) * 26), domain) == (IT(2), IT(3))
        @test Class.inside(FT(1.5), FT(0.3), domain) == true
        @test Class.inside(FT(-1.5), FT(0.3), domain) == false
        @test Class.indexCartesianFromPosition(FT(1.5), FT(0.3), domain) == (IT(17), IT(16))
        @test Class.indexLinearFromPosition(FT(1.5), FT(0.3), domain) == (IT(17) + (IT(16) - 1) * 26)
    end
    @testset "Domain3D" begin
        x_0 = 0
        y_0 = 0
        z_0 = 0
        x_1 = 0.4
        y_1 = 0.5
        z_1 = 0.7
        gap = 0.15
        domain = Class.Domain3D{IT, FT}(gap, x_0, y_0, z_0, x_1, y_1, z_1)
        @test Class.dimension(domain) == 3
        @test Class.get_gap(domain) ≈ gap
        @test Class.get_gap_square(domain) ≈ gap * gap
        @test Class.get_n_x(domain) == 2
        @test Class.get_n_y(domain) == 3
        @test Class.get_n_z(domain) == 4
        @test Class.get_n_xy(domain) == 2 * 3
        @test Class.get_n_yz(domain) == 3 * 4
        @test Class.get_n_zx(domain) == 4 * 2
        @test Class.get_n(domain) == 2 * 3 * 4
        @test Class.get_first_x(domain) ≈ x_0
        @test Class.get_first_y(domain) ≈ y_0
        @test Class.get_first_z(domain) ≈ z_0
        @test Class.get_last_x(domain) ≈ x_1
        @test Class.get_last_y(domain) ≈ y_1
        @test Class.get_last_z(domain) ≈ z_1
        @test Class.get_span_x(domain) ≈ x_1 - x_0
        @test Class.get_span_y(domain) ≈ y_1 - y_0
        @test Class.get_span_z(domain) ≈ z_1 - z_0
        @test Class.get_gap_x(domain) ≈ (x_1 - x_0) / Class.get_n_x(domain)
        @test Class.get_gap_y(domain) ≈ (y_1 - y_0) / Class.get_n_y(domain)
        @test Class.get_gap_z(domain) ≈ (z_1 - z_0) / Class.get_n_z(domain)
        @test Class.get_gap_x_inv(domain) ≈ Class.get_n_x(domain) / (x_1 - x_0)
        @test Class.get_gap_y_inv(domain) ≈ Class.get_n_y(domain) / (y_1 - y_0)
        @test Class.get_gap_z_inv(domain) ≈ Class.get_n_z(domain) / (z_1 - z_0)

        @test Class.indexCartesianToLinear(IT(1), IT(2), IT(3), domain) == IT(15)
        @test Class.indexLinearToCartesian(IT(15), domain) == (IT(1), IT(2), IT(3))
        @test Class.inside(FT(0.1), FT(0.2), FT(0.3), domain) == true
        @test Class.inside(FT(0.5), FT(0.2), FT(0.3), domain) == false
        @test Class.indexCartesianFromPosition(FT(0.3), FT(0.32), FT(0.36), domain) == (IT(2), IT(2), IT(3))
        @test Class.indexLinearFromPosition(FT(0.3), FT(0.32), FT(0.36), domain) == 2 + (2 - 1) * 2 + (3 - 1) * 2 * 3
    end
end
