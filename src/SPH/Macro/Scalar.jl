#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/12 21:40:32
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

# * use in scope with `INDEX` and `INT`
const kIntScalarDict = Dict("Tag" => ["tag"], "IsMovable" => ["is_movable", "not_fixed"], "nCount" => ["count"])
# * use in scope with `INDEX` and `FLOAT`
const kFloatScalarDict = Dict(
    "Mass" => ["mass", "m"],
    "Volume" => ["vol", "volume"],
    "Density" => ["rho", "ρ", "density"],
    "dDensity" => ["drho", "dρ", "density_ratio"],
    "Pressure" => ["p", "pressure", "P"],
    "Gap" => ["gap", "Δx", "Δp"],
    "H" => ["h"],
    "SumWeight" => ["wv", "∑wv"], # ∑wᵢⱼVⱼ
    "SumWeightedDensity" => ["wv_rho", "∑wv_rho", "wv_ρ", "∑wv_ρ"], # ∑wᵢⱼVⱼρⱼ
    "SumWeightedPressure" => ["wv_p", "∑wv_p"], # ∑wᵢⱼVⱼPⱼ
)
# * use in scope with `INDEX` and `NI` and `INT`
const kNeighbourIntScalarDict = Dict("nIndex" => ["n_index"])
# * use in scope with `INDEX` and `NI` and `FLOAT`
const kNeighbourFloatScalarDict = Dict(
    "nR" => ["r", "n_r"],
    "nW" => ["w", "n_w"],
    "nDW" => ["dw", "∇w", "n_dw", "n_∇w"],
    "nHInv" => ["hinv", "n_hinv", "h_inv"],
)

# """
# eg. 
# - `@Tag` return `INDEX.Tag`
# - `@Tag(i)` return `INT[i, INDEX.Tag]`
# """
for key in keys(kIntScalarDict)
    names = kIntScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(INDEX, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(i)
            return esc(:(INT[\$i, INDEX.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@Mass` return `INDEX.Mass`
# - `@Mass(i)` return `FLOAT[i, INDEX.Mass]`
# """
for key in keys(kFloatScalarDict)
    names = kFloatScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(getfield(INDEX, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(i)
            return esc(:(FLOAT[\$i, INDEX.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nIndex` return `NI + INDEX.nIndex`
# - `@nIndex(ni)` return `INT[I, ni + INDEX.nIndex]` which is `J`
# """
for key in keys(kNeighbourIntScalarDict)
    names = kNeighbourIntScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(NI + getfield(INDEX, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(ni)
            return esc(:(INT[I, \$ni + INDEX.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end

# """
# eg.
# - `@nR` return `NI + INDEX.nR`
# - `@nR(ni)` return `FLOAT[I, ni + INDEX.nR]`
# """
for key in keys(kNeighbourFloatScalarDict)
    names = kNeighbourFloatScalarDict[key]
    push!(names, key)
    for name in names
        eval(Meta.parse("""
                        macro $name()
                            return esc(:(NI + getfield(INDEX, :$key)))
                        end
                        """))
        eval(Meta.parse("""
        macro $name(ni)
            return esc(:(FLOAT[I, \$ni + INDEX.$key]))
        end
        """))
        eval(Meta.parse("export @$name"))
    end
end
