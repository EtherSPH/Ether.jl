#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 20:44:08
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Kernel" begin
    Pkg.add("SPHKernels")
    using SPHKernels
    # * see reference [url](https://github.com/LudwigBoess/SPHKernels.jl/tree/dccf7fd6e34215428e145639e6c1c456f7c4ef10)
    for dim in [2, 3]
        for h in FT.(0.05:0.05:0.2)
            for x in FT.(0.02:0.02:0.1)
                my_ck = SPH.Kernel.CubicSpline{IT, FT, dim}()
                sp_ck = SPHKernels.Cubic(FT, dim)
                @test SPH.Kernel.value(x, h, my_ck) ≈ SPHKernels.kernel_value(sp_ck, x / (2 * h), 1 / (2 * h))
                @test SPH.Kernel.gradient(x, h, my_ck) ≈ SPHKernels.kernel_deriv(sp_ck, x / (2 * h), 1 / (2 * h))
                my_w2k = SPH.Kernel.WendlandC2{IT, FT, dim}()
                sp_w2k = SPHKernels.WendlandC2(FT, dim)
                @test SPH.Kernel.value(x, h, my_w2k) ≈ SPHKernels.kernel_value(sp_w2k, x / (2 * h), 1 / (2 * h))
                @test SPH.Kernel.gradient(x, h, my_w2k) ≈ SPHKernels.kernel_deriv(sp_w2k, x / (2 * h), 1 / (2 * h))
                my_w4k = SPH.Kernel.WendlandC4{IT, FT, dim}()
                sp_w4k = SPHKernels.WendlandC4(FT, dim)
                @test SPH.Kernel.value(x, h, my_w4k) ≈ SPHKernels.kernel_value(sp_w4k, x / (2 * h), 1 / (2 * h))
                # ! seems that `SPHKernels.jl` makes some mistakes
                # ! see [url](https://github.com/LudwigBoess/SPHKernels.jl/blob/dccf7fd6e34215428e145639e6c1c456f7c4ef10/src/wendland/C4.jl)
                @test abs(
                    SPH.Kernel.gradient(x, h, my_w4k) - SPHKernels.kernel_deriv(sp_w4k, x / (2 * h), 1 / (2 * h)),
                ) / abs(SPH.Kernel.gradient(h * FT(0.7), h, my_w4k)) < 2e-2
            end
        end
    end
end
