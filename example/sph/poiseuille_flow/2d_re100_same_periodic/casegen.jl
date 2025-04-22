#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/22 15:35:26
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
const FT = Float64
const CT = Array
const Backend = KernelAbstractions.CPU()
const parallel = Environment.Parallel{IT, FT, CT, Backend}()
@info parallel

# * ===================== domain ===================== * #

const x0 = 0.0
const y0 = 0.0
const pipe_width = 1e-2
const pipe_length = 4 * pipe_width

const particle_gap = pipe_width / 50
const domain_gap = 3 * particle_gap
const domain = Class.Domain2D{IT, FT}(domain_gap, x0, y0 - domain_gap, x0 + pipe_length, y0 + pipe_width + domain_gap)
const dimension = Class.dimension(domain)
const Dimension = Environment.Dimension2D

@info domain

# * ===================== particle_system ===================== * #

const FLUID_TAG = 1
const WALL_TAG = 2

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
)

const reynolds = 100.0
const referencelength = pipe_width
const rho0 = 1e3
const mu0 = 1e-3
const umean = reynolds * mu0 / rho0 / referencelength
const umax = 1.5 * umean
const fx = 12 * mu0 * umean / pipe_width^2
const ax = fx / rho0
const gx = ax
const gy = 0.0
const c0 = 10 * umax
const p0 = 0.01 * rho0 * c0^2
const wall_width = domain_gap

const parameters = (
    rho0 = rho0,
    p0 = p0,
    FLUID_TAG = FLUID_TAG,
    WALL_TAG = WALL_TAG,
    gap0 = particle_gap,
    h0 = 1.5 * particle_gap,
    mu0 = mu0,
    reynolds = reynolds,
    c0 = c0,
    fx = fx,
    gx = gx,
    gy = gy,
    umean = umean,
    umax = umax,
    pipe_width = pipe_width,
)

# * fluid particles
function modifyFluid!(ps)
    ps[:Tag] = FLUID_TAG
    ps[:IsMovable] = 1
    ps[:Density] = rho0
    ps[:Mass] = ps[:Density] .* ps[:Volume]
    ps[:H] = ps[:Gap] * 1.5
    ps[:Pressure] = p0
    return nothing
end

const fluid_column = Geometry.Rectangle{FT}(x0, y0, x0 + pipe_length, y0 + pipe_width)
const n_fluid = Geometry.count(particle_gap, fluid_column)
fluid_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_fluid, int_named_tuple, float_named_tuple)
positions, volumes, gaps = Geometry.create(particle_gap, fluid_column; parallel = true)
Class.count!(fluid_particles, n_fluid)
fluid_particles[:PositionVec] = positions
fluid_particles[:Volume] = volumes
fluid_particles[:Gap] = gaps
modifyFluid!(fluid_particles)

# * wall particles
function modifyWall!(ps)
    ps[:Tag] = WALL_TAG
    ps[:IsMovable] = 0
    ps[:Density] = rho0
    ps[:Mass] = ps[:Density] .* ps[:Volume]
    ps[:H] = ps[:Gap] * 1.5
    ps[:Pressure] = p0
    return nothing
end
# * bottom wall particles
const bottom_wall = Geometry.Rectangle{FT}(x0, y0 - wall_width, x0 + pipe_length, y0)
const n_bottom_wall = Geometry.count(particle_gap, bottom_wall)
bottom_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_bottom_wall, int_named_tuple, float_named_tuple)
Class.count!(bottom_wall_particles, n_bottom_wall)
positions, volumes, gaps = Geometry.create(particle_gap, bottom_wall; parallel = true)
bottom_wall_particles[:PositionVec] = positions
bottom_wall_particles[:Volume] = volumes
bottom_wall_particles[:Gap] = gaps
modifyWall!(bottom_wall_particles)
# * top wall particles
const top_wall = Geometry.Rectangle{FT}(x0, y0 + pipe_width, x0 + pipe_length, y0 + pipe_width + wall_width)
const n_top_wall = Geometry.count(particle_gap, top_wall)
top_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_top_wall, int_named_tuple, float_named_tuple)
Class.count!(top_wall_particles, n_top_wall)
positions, volumes, gaps = Geometry.create(particle_gap, top_wall; parallel = true)
top_wall_particles[:PositionVec] = positions
top_wall_particles[:Volume] = volumes
top_wall_particles[:Gap] = gaps
modifyWall!(top_wall_particles)

particle_system = merge(fluid_particles, bottom_wall_particles, top_wall_particles)
Class.set_is_alive!(particle_system)

@info particle_system

# * ===================== neighbour_system ===================== * #

const active_pair = [FLUID_TAG => FLUID_TAG, FLUID_TAG => WALL_TAG, WALL_TAG => FLUID_TAG]
const periodic_boundary = Class.PeriodicBoundary2D{true, false}

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

writer = DataIO.Writer(path_ = "example/result/sph/poiseuille_flow/2d_re100_same_periodic")
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
