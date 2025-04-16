#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 21:40:38
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * use in scope with `INDEX` and `FLOAT`
const kFloatVectorDict = Dict(
    "PositionVec" => ["x", "Position", "position"],
    "VelocityVec" => ["u", "Velocity", "velocity"],
    "dVelocityVec" => ["du", "dVelocity", "dvelocity"],
    "AccelerationVec" => ["a", "Acceleration", "acceleration"],
)
# * use in scope with `INDEX` and `NI` and `FLOAT`
const kNeighbourFloatVectorDict = Dict("nRVec" => ["n_rvec", "rvec"])

# """
# eg.
# - `@VelocityVec` return `INDEX.VelocityVec`
# - `@VelocityVec(index, i)` return `FLOAT[index, INDEX.VelocityVec + i]`
# """
for key in keys(kFloatVectorDict)
    names = kFloatVectorDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(INDEX, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(index, i)
            return esc(:(FLOAT[\$index, INDEX.$key + \$i]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nRVec` return `NI + INDEX.nRVec`
# - `@nR(ni, i)` return `FLOAT[I, dimension * ni + INDEX.nR + i]`
# """
for key in keys(kNeighbourFloatVectorDict)
    names = kNeighbourFloatVectorDict[key]
    push!(names, key)
    for name in names
        eval(
            Meta.parse(
                """
                macro $name()
                    return esc(:(Environment.capacity(Val(DIMENSION), Val(Environment.Tvector)) * NI + getfield(INDEX, :$key)))
                end
                """,
            ),
        )
        eval(
            Meta.parse(
                """
macro $name(ni, i)
    return esc(:(FLOAT[I, (Environment.capacity(Val(DIMENSION), Val(Environment.Tvector)) * \$ni) + INDEX.$key + \$i]))
end
""",
            ),
        )
        eval(Meta.parse("export @$name"))
    end
end
