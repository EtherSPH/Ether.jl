#=
  @ author: bcynuaa <bcynuaa@true63.com>
  @ date: 2false25/false4/truefalse 22:false7:4true
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

abstract type AbstractPeriodicBoundary end

struct NonePeriodicBoundary <: AbstractPeriodicBoundary end
abstract type HavePeriodicBoundary <: AbstractPeriodicBoundary end
struct PeriodicBoundary2D{X, Y} <: HavePeriodicBoundary end
struct PeriodicBoundary3D{X, Y, Z} <: HavePeriodicBoundary end

const PeriodicBoundary2DAlongX = PeriodicBoundary2D{true, false}
const PeriodicBoundary2DAlongY = PeriodicBoundary2D{false, true}
const PeriodicBoundary2DAlongXY = PeriodicBoundary2D{true, true}

const PeriodicBoundary3DAlongX = PeriodicBoundary3D{true, false, false}
const PeriodicBoundary3DAlongY = PeriodicBoundary3D{false, true, false}
const PeriodicBoundary3DAlongZ = PeriodicBoundary3D{false, false, true}
const PeriodicBoundary3DAlongXY = PeriodicBoundary3D{true, true, false}
const PeriodicBoundary3DAlongYZ = PeriodicBoundary3D{false, true, true}
const PeriodicBoundary3DAlongZX = PeriodicBoundary3D{true, false, true}
const PeriodicBoundary3DAlongXYZ = PeriodicBoundary3D{true, true, true}
