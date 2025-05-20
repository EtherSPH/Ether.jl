#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/20 15:28:05
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
const z0 = 0.0
const box_x_len = 0.42
const box_y_len = 0.228
const box_z_len = 0.44
const fluid_x_len = 0.114
const fluid_y_len = 0.228
const fluid_z_len = 0.114
const particle_gap = fluid_x_len / 32
const h_ratio = 1.0
const wall_width = h_ratio * particle_gap * 2
const domain_gap = wall_width
const domain = Class.Domain3D{IT, FT}(
    domain_gap,
    x0 - domain_gap,
    y0 - domain_gap,
    z0 - domain_gap,
    x0 + box_x_len + domain_gap,
    y0 + box_y_len + domain_gap,
    z0 + box_z_len + domain_gap,
)
const dimension = Class.dimension(domain)
const Dimension = Environment.Dimension3D
@info domain

# * ===================== particle_system ===================== * #

const FLUID_TAG = 1
const WALL_TAG = 2

const max_neighbour_number = 80
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
const parameters = (
    rho0 = 1e3,
    p0 = 0.0,
    FLUID_TAG = FLUID_TAG,
    WALL_TAG = WALL_TAG,
    gap0 = particle_gap,
    h0 = h_ratio * particle_gap,
    mu0 = 1e-3,
    fluid_x_len = fluid_x_len,
    fluid_y_len = fluid_y_len,
    fluid_z_len = fluid_z_len,
)

# * fluid particles
function modifyFluid!(ps)
    ps[:Tag] = FLUID_TAG
    ps[:IsMovable] = 1
    ps[:Density] = parameters.rho0
    ps[:Mass] = ps[:Density] .* ps[:Volume]
    ps[:H] = ps[:Gap] * h_ratio
    ps[:Pressure] = parameters.p0
    return nothing
end

# * wall particles
function modifyWall!(ps)
    ps[:Tag] = WALL_TAG
    ps[:IsMovable] = 0
    ps[:Density] = parameters.rho0
    ps[:Mass] = ps[:Density] .* ps[:Volume]
    ps[:H] = ps[:Gap] * h_ratio
    ps[:Pressure] = parameters.p0
    return nothing
end

const fluid_shape = Geometry.Cuboid{FT}(x0, y0, z0, x0 + fluid_x_len, y0 + fluid_y_len, z0 + fluid_z_len)
const n_fluid_shape = Geometry.count(particle_gap, fluid_shape)
fluid_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_fluid_shape, int_named_tuple, float_named_tuple)
Class.count!(fluid_particles, n_fluid_shape)
positions, volumes, gaps = Geometry.create(particle_gap, fluid_shape; parallel = true)
fluid_particles[:PositionVec] = positions
fluid_particles[:Volume] = volumes
fluid_particles[:Gap] = gaps
modifyFluid!(fluid_particles)

const bottom_wall_shape = Geometry.Cuboid{FT}(
    x0 - wall_width,
    y0 - wall_width,
    z0 - wall_width,
    x0 + box_x_len + wall_width,
    y0 + box_y_len + wall_width,
    z0,
)
const n_bottom_wall_shape = Geometry.count(particle_gap, bottom_wall_shape)
bottom_wall_particles =
    Class.HostParticleSystem{IT, FT, Dimension}(n_bottom_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(bottom_wall_particles, n_bottom_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, bottom_wall_shape; parallel = true)
bottom_wall_particles[:PositionVec] = positions
bottom_wall_particles[:Volume] = volumes
bottom_wall_particles[:Gap] = gaps
modifyWall!(bottom_wall_particles)

const back_wall_shape = Geometry.Cuboid{FT}(
    x0 - wall_width,
    y0 + box_y_len,
    z0,
    x0 + box_x_len + wall_width,
    y0 + box_y_len + wall_width,
    z0 + box_z_len + wall_width,
)
const n_back_wall_shape = Geometry.count(particle_gap, back_wall_shape)
back_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_back_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(back_wall_particles, n_back_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, back_wall_shape; parallel = true)
back_wall_particles[:PositionVec] = positions
back_wall_particles[:Volume] = volumes
back_wall_particles[:Gap] = gaps
modifyWall!(back_wall_particles)

const fornt_wall_shape = Geometry.Cuboid{FT}(
    x0 - wall_width,
    y0 - wall_width,
    z0,
    x0 + box_x_len + wall_width,
    y0,
    z0 + box_z_len + wall_width,
)
const n_fornt_wall_shape = Geometry.count(particle_gap, fornt_wall_shape)
fornt_wall_particles =
    Class.HostParticleSystem{IT, FT, Dimension}(n_fornt_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(fornt_wall_particles, n_fornt_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, fornt_wall_shape; parallel = true)
fornt_wall_particles[:PositionVec] = positions
fornt_wall_particles[:Volume] = volumes
fornt_wall_particles[:Gap] = gaps
modifyWall!(fornt_wall_particles)

left_wall_shape = Geometry.Cuboid{FT}(x0 - wall_width, y0, z0, x0, y0 + box_y_len, z0 + box_z_len + wall_width)
const n_left_wall_shape = Geometry.count(particle_gap, left_wall_shape)
left_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_left_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(left_wall_particles, n_left_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, left_wall_shape; parallel = true)
left_wall_particles[:PositionVec] = positions
left_wall_particles[:Volume] = volumes
left_wall_particles[:Gap] = gaps
modifyWall!(left_wall_particles)

right_wall_shape = Geometry.Cuboid{FT}(
    x0 + box_x_len,
    y0,
    z0,
    x0 + box_x_len + wall_width,
    y0 + box_y_len,
    z0 + box_z_len + wall_width,
)
const n_right_wall_shape = Geometry.count(particle_gap, right_wall_shape)
right_wall_particles =
    Class.HostParticleSystem{IT, FT, Dimension}(n_right_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(right_wall_particles, n_right_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, right_wall_shape; parallel = true)
right_wall_particles[:PositionVec] = positions
right_wall_particles[:Volume] = volumes
right_wall_particles[:Gap] = gaps
modifyWall!(right_wall_particles)

top_wall_shape =
    Geometry.Cuboid{FT}(x0, y0, z0 + box_z_len, x0 + box_x_len, y0 + box_y_len, z0 + box_z_len + wall_width)
const n_top_wall_shape = Geometry.count(particle_gap, top_wall_shape)
top_wall_particles = Class.HostParticleSystem{IT, FT, Dimension}(n_top_wall_shape, int_named_tuple, float_named_tuple)
Class.count!(top_wall_particles, n_top_wall_shape)
positions, volumes, gaps = Geometry.create(particle_gap, top_wall_shape; parallel = true)
top_wall_particles[:PositionVec] = positions
top_wall_particles[:Volume] = volumes
top_wall_particles[:Gap] = gaps
modifyWall!(top_wall_particles)

wall_particles = merge(
    bottom_wall_particles,
    back_wall_particles,
    fornt_wall_particles,
    left_wall_particles,
    right_wall_particles,
    top_wall_particles,
)
Class.set_is_alive!(wall_particles)

particle_system = merge(fluid_particles, wall_particles)
Class.set_is_alive!(particle_system)

@info particle_system

# * ===================== neighbour_system ===================== * #

const active_pair = [FLUID_TAG => FLUID_TAG, FLUID_TAG => WALL_TAG, WALL_TAG => FLUID_TAG]
const periodic_boundary = Class.NonePeriodicBoundary

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

writer = DataIO.Writer(joinpath(@__DIR__, "../../../result/sph/cruchaga/3d_extrapolation"))
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
