#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/26 21:18:34
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using OrderedCollections
using KernelAbstractions

using Ether

config_dict = DataIO.template()

# * ===================== parallel ===================== * #

const IT = Int32
const FT = Float32
const CT = Array
const Backend = KernelAbstractions.CPU()
const parallel = Environment.Parallel{IT, FT, CT, Backend}()
@info parallel

# * ===================== domain ===================== * #

const x0 = 0.0
const y0 = 0.0
const r_outer = 1.0
const radius_ratio = 2.6
const r_inner = r_outer / radius_ratio
const particle_gap = 0.01
const domain_gap = particle_gap * 3
const wall_width = domain_gap
const domain = Class.Domain2D{IT, FT}(
    domain_gap,
    x0 - r_outer - wall_width,
    y0 - r_outer - wall_width,
    x0 + r_outer + wall_width,
    y0 + r_outer + wall_width,
)
const dimension = Class.dimension(domain)
const Dimension = Environment.Dimension2D
@info domain

# * ===================== particle_system ===================== * #

const FLUID_TAG = 1
const THERMOSTATIC_WALL_TAG = 2
const max_neighbour_number = 40
const int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = max_neighbour_number)
const float_named_tuple = (
    PositionVec = dimension,
    VelocityVec = dimension,
    dVelocityVec = dimension,
    AccelerationVec = dimension,
    Mass = 1,
    Volume = 1,
    Density = 1,
    dDensity = 1,
    Pressure = 1,
    Gap = 1,
    H = 1,
    SumWeight = 1,
    SumWeightedDensity = 1,
    SumWeightedPressure = 1,
    nW = max_neighbour_number,
    nDW = max_neighbour_number,
    nHInv = max_neighbour_number,
    nRVec = max_neighbour_number * dimension,
    nR = max_neighbour_number,
    NormalVec = dimension,
    Temperature = 1,
    dTemperature = 1,
)
const reference_length = r_outer - r_inner
const prandtl = 0.71
const rayleigh = 4.7e4
const c0 = 2.0
const beta = 0.05
const gravity = 1.0
const mu0 = 1e-3
const kappa = 1.0

const touter = 0.0
const tinner = 1.0
const tdelta = tinner - touter
const tmean = 0.5 * (touter + tinner)
const t0 = 0.2 * tdelta + touter # 0.2 is a magic number which can be adjusted

const cp = prandtl * kappa / mu0
const rho0 = sqrt(rayleigh * mu0 * kappa / gravity / beta / reference_length^3 / tdelta / cp)
const p0 = 0.02 * rho0 * c0^2
const alpha = mu0 / prandtl / rho0
const parameters = (
    rho0 = rho0,
    p0 = p0,
    gap0 = particle_gap,
    h0 = 1.5 * particle_gap,
    FLUID_TAG = FLUID_TAG,
    THERMOSTATIC_WALL_TAG = THERMOSTATIC_WALL_TAG,
    reference_length = reference_length,
    mu0 = mu0,
    c0 = c0,
    kappa = kappa,
    cp = cp,
    alpha = alpha,
    beta = beta,
    gravity = gravity,
    t0 = t0,
)

const fluid_shape = Geometry.Ring{FT}(x0, y0, r_inner, r_outer)
const inner_wall_shape = Geometry.Ring{FT}(x0, y0, r_inner - wall_width, r_inner)
const outer_wall_shape = Geometry.Ring{FT}(x0, y0, r_outer, r_outer + wall_width)
@inline function normalize(x, y)
    r = sqrt(x * x + y * y)
    return x / r, y / r
end
function modify!(ps, i)
    ps[:Density, i] = rho0
    ps[:Mass, i] = ps[:Density, i] * ps[:Volume, i]
    ps[:H, i] = ps[:Gap, i] * 1.5
    ps[:Pressure, i] = p0
    x, y = ps[:PositionVec, i]
    if Geometry.inside(x, y, fluid_shape)
        # fluid particles
        ps[:Tag, i] = FLUID_TAG
        ps[:Temperature, i] = t0
        ps[:IsMovable, i] = 1
    elseif Geometry.inside(x, y, inner_wall_shape)
        # inner wall particles
        ps[:Tag, i] = THERMOSTATIC_WALL_TAG
        ps[:IsMovable, i] = 0
        ps[:Temperature, i] = tinner
        x_, y_ = normalize(x - x0, y - y0)
        ps[:NormalVec, i] = [x_, y_]
    elseif Geometry.inside(x, y, outer_wall_shape)
        # outer wall particles
        ps[:Tag, i] = THERMOSTATIC_WALL_TAG
        ps[:IsMovable, i] = 0
        ps[:Temperature, i] = touter
        x_, y_ = normalize(x - x0, y - y0)
        ps[:NormalVec, i] = [-x_, -y_]
    else
        error("Particle position is out of bounds: ($x, $y) at index $i")
    end
end

const n_fluid = Geometry.count(particle_gap, fluid_shape)
fluid_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_fluid, int_named_tuple, float_named_tuple)
Class.count!(fluid_particles, n_fluid)
positions, volumes, gaps = Geometry.create(particle_gap, fluid_shape)
fluid_particles[:PositionVec] = positions
fluid_particles[:Volume] = volumes
fluid_particles[:Gap] = gaps
Threads.@threads for i in 1:n_fluid
    modify!(fluid_particles, i)
end

const n_inner_wall = Geometry.count(particle_gap, inner_wall_shape)
inner_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_inner_wall, int_named_tuple, float_named_tuple)
Class.count!(inner_wall_particles, n_inner_wall)
positions, volumes, gaps = Geometry.create(particle_gap, inner_wall_shape)
inner_wall_particles[:PositionVec] = positions
inner_wall_particles[:Volume] = volumes
inner_wall_particles[:Gap] = gaps
Threads.@threads for i in 1:n_inner_wall
    modify!(inner_wall_particles, i)
end

const n_outer_wall = Geometry.count(particle_gap, outer_wall_shape)
outer_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_outer_wall, int_named_tuple, float_named_tuple)
Class.count!(outer_wall_particles, n_outer_wall)
positions, volumes, gaps = Geometry.create(particle_gap, outer_wall_shape)
outer_wall_particles[:PositionVec] = positions
outer_wall_particles[:Volume] = volumes
outer_wall_particles[:Gap] = gaps
Threads.@threads for i in 1:n_outer_wall
    modify!(outer_wall_particles, i)
end

particle_system = merge(fluid_particles, inner_wall_particles, outer_wall_particles)
Class.set_is_alive!(particle_system)
@info particle_system

# * ===================== neighbour_system ===================== * #

const active_pair = [FLUID_TAG => FLUID_TAG, FLUID_TAG => THERMOSTATIC_WALL_TAG]
const periodic_boundary = Class.PeriodicBoundary2D{false, false}

neighbour_system = Class.NeighbourSystem(
    periodic_boundary,
    parallel,
    domain,
    active_pair;
    max_neighbour_number = max_neighbour_number,
    n_threads = 32,
)
@info neighbour_system

# * ===================== writer ===================== * #

writer = DataIO.Writer(joinpath(@__DIR__, "../../../result/sph/thermal_ring/ra4.7e4_ring2.6_compulsive"))
DataIO.rmdir(writer)
DataIO.mkdir(writer)

replace!(config_dict, parallel)
replace!(config_dict, domain)
replace!(config_dict, particle_system)
replace!(config_dict, neighbour_system)
replace!(config_dict, writer)
config_dict["parameters"] = Utility.namedtuple2dict(parameters; dicttype = OrderedDict)
DataIO.save(config_dict)
appendix = DataIO.appendix()
appendix["WriteStep"] = 0
appendix["TimeValue"] = 0.0
appendix["TMSTEP"] = 0
DataIO.save!(writer, particle_system, appendix)
DataIO.wait!(writer)
