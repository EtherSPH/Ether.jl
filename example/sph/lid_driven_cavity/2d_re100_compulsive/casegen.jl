#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/19 17:48:37
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
const cavity_length = 1.0
const particle_gap = cavity_length / 100
const domain_gap = 3 * particle_gap
const wall_width = 3 * particle_gap
const domain = Class.Domain2D{IT, FT}(
    domain_gap,
    x0 - domain_gap,
    y0 - domain_gap,
    x0 + cavity_length + domain_gap,
    y0 + cavity_length + domain_gap,
)
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
    NormalVec = dimension,
)
const reynolds = 100.0
const reference_length = cavity_length
const rho0 = 1.0
const reference_velocity = 1.0
const mu0 = rho0 * reference_velocity * reference_length / reynolds
const c0 = 10 * reference_velocity
const p0 = c0 * c0 * rho0 * 0.005
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
    cavity_length = cavity_length,
    reference_length = reference_length,
)

const fluid_column = Geometry.Rectangle{FT}(x0, y0, x0 + cavity_length, y0 + cavity_length)
const calculation_column = Geometry.Rectangle{FT}(
    x0 - wall_width,
    y0 - wall_width,
    x0 + cavity_length + wall_width,
    y0 + cavity_length + wall_width,
)

function modify!(ps, i)
    ps[:Density, i] = rho0
    ps[:Mass, i] = ps[:Density, i] * ps[:Volume, i]
    ps[:H, i] = ps[:Gap, i] * 1.5
    ps[:Pressure, i] = p0
    x, y = ps[:PositionVec, i]
    if Geometry.inside(x, y, fluid_column)
        # fluid particles
        ps[:Tag, i] = FLUID_TAG
        ps[:IsMovable, i] = 1
    else
        # wall particles
        ps[:Tag, i] = WALL_TAG
        ps[:IsMovable, i] = 0
        if y < y0
            if x0 < x < x0 + cavity_length
                ps[:NormalVec, i] = [0.0, 1.0]
            elseif x < x0
                ps[:NormalVec, i] = [1.0, 1.0] ./ sqrt(2)
            elseif x > x0 + cavity_length
                ps[:NormalVec, i] = [-1.0, 1.0] ./ sqrt(2)
            end
        elseif y > y0 + cavity_length
            ps[:VelocityVec, i] = [reference_velocity, 0.0]
            if x0 < x < x0 + cavity_length
                ps[:NormalVec, i] = [0.0, -1.0]
            elseif x < x0
                ps[:NormalVec, i] = [1.0, -1.0] ./ sqrt(2)
            elseif x > x0 + cavity_length
                ps[:NormalVec, i] = [-1.0, -1.0] ./ sqrt(2)
            end
        else
            if x < x0
                ps[:NormalVec, i] = [1.0, 0.0]
            elseif x > x0 + cavity_length
                ps[:NormalVec, i] = [-1.0, 0.0]
            end
        end
    end
end

const n_particles = Geometry.count(particle_gap, calculation_column)
particle_system = Class.HostParticleSystem{IT, FT, Dimension}(n_particles, int_named_tuple, float_named_tuple)
Class.count!(particle_system, n_particles)
positions, volumes, gaps = Geometry.create(particle_gap, calculation_column; parallel = true)
particle_system[:PositionVec] = positions
particle_system[:Volume] = volumes
particle_system[:Gap] = gaps
Threads.@threads for i in 1:n_particles
    modify!(particle_system, i)
end

@info particle_system

# * ===================== neighbour_system ===================== * #

const active_pair = [FLUID_TAG => FLUID_TAG, FLUID_TAG => WALL_TAG, WALL_TAG => FLUID_TAG]
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

writer = DataIO.Writer(joinpath(@__DIR__, "../../../result/sph/lid_driven_cavity/2d_re100_compulsive"))
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
