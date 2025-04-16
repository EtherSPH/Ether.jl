#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/14 15:18:23
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "UtilityTest" begin
    time_format = "yyyy_mm_dd_HH:MM:SS"
    time_str = Utility.timestamp(format = time_format)
    @test time_str isa String
    @test length(time_str) == length(time_format)

    @test Utility.namedtuple2dict((a = 1, b = 2); dicttype = Dict, keytype = Symbol) == Dict(:a => 1, :b => 2)
    @test Utility.dict2namedtuple(OrderedDict("a" => 1, "b" => 2)) == (a = 1, b = 2)
end
