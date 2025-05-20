#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/19 19:53:14
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using JSON
using OrderedCollections
using KernelAbstractions
using ProgressMeter

using Ether
using Ether.Macro
using Ether.SPH.Macro

config_dict = JSON.parsefile(
    joinpath(@__DIR__, "../../../result/sph/lid_driven_cavity/2d_re100_compulsive/config/config.json");
    dicttype = OrderedDict,
)
# * cpu
# config_dict["parallel"]["backend"] = "cpu"
# const n_threads = 30
# * cuda
using CUDA
config_dict["parallel"]["backend"] = "cuda"
const n_threads = 1024
# * rocm
# using AMDGPU
# config_dict["parallel"]["backend"] = "rocm"
# const n_threads = 1024
# * oneapi
# using oneAPI
# config_dict["parallel"]["backend"] = "oneapi"
# const n_threads = 256
# * metal
# using Metal
# config_dict["parallel"]["backend"] = "metal"
# const n_threads = 256

const backend = config_dict["parallel"]["backend"]
const color = Environment.kNameToColor[backend]

printstyled("running on $backend\n", color = color)

# * ===================== parallel ===================== * #
DataIO.ParallelExpr(config_dict) |> eval
# * ===================== domain ===================== * #
const domain = DataIO.Domain(config_dict)
const IT = typeof(domain).parameters[1]
const FT = typeof(domain).parameters[2]
const dimension = Class.dimension(domain)
const Dimension = supertypes(typeof(domain))[3].parameters[3]
# * ===================== particle_system ===================== * #
dps = DataIO.ParticleSystem(config_dict, parallel) # device particle system
hps = Class.mirror(dps) # host particle system
# * ===================== neighbour_system ===================== * #
ns = DataIO.NeighbourSystem(config_dict, parallel)
# * ===================== writer ===================== * #
writer = DataIO.Writer(config_dict)
# * ===================== parameters ===================== * #
const parameters = Utility.dict2namedtuple(config_dict["parameters"])
const FLUID_TAG = parameters.FLUID_TAG |> parallel
const WALL_TAG = parameters.WALL_TAG |> parallel
const rho0 = parameters.rho0 |> parallel
const p0 = parameters.p0 |> parallel
const gap0 = parameters.gap0 |> parallel
const h0 = parameters.h0 |> parallel
const mu0 = parameters.mu0 |> parallel
const mu0_2 = 2 * mu0 |> parallel
const c0 = parameters.c0 |> parallel
const c02 = c0 * c0 |> parallel
const sph_kernel = SPH.Kernel.CubicSpline{IT, FT, Int(dimension)}()

const total_time = parallel(20.0)
const total_time_inv = parallel(1 / total_time)
const dt = parallel(0.2 * h0 / c0)
const output_interval = 200
const filter_interval = 20

@inline eos(rho::Real) = c02 * (rho - rho0) + p0

# * ===================== particle action definition ===================== * #

@inline function sComputeKernelAndiContinuity!(@self_args)::Nothing
    @with_neighbours @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
        SPH.Library.iValueGradient!(@inter_args, sph_kernel)
        SPH.Library.iClassicContinuity!(@inter_args; dw = @dw(@ij))
    end
    return nothing
end

@inline function sContinuity!(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sContinuity!(@self_args; dt = dt)
        SPH.Library.sVolume!(@self_args)
        @inbounds @p(@i) = eos(@rho(@i))
        return nothing
    end
    return nothing
end

@inline function iMomentum!(@inter_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
        SPH.Library.iClassicPressure!(@inter_args; dw = @dw(@ij))
        SPH.Library.iClassicViscosity!(@inter_args; dw = @dw(@ij), mu = mu0)
        return nothing
    elseif @tag(@i) == FLUID_TAG && @tag(@j) == WALL_TAG
        SPH.Library.iClassicPressure!(@inter_args; dw = @dw(@ij))
        SPH.Library.iClassicViscosity!(@inter_args; dw = @dw(@ij), mu = mu0)
        SPH.Library.iCompulsive!(@inter_args; c = c0, h = h0)
        return nothing
    end
    return nothing
end

@inline function sMomentum!(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sAccelerateMove!(@self_args; dt = dt)
    end
    return nothing
end

@inline function iFilter!(@inter_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG && @tag(@j) == FLUID_TAG
        SPH.Library.iKernelFilter!(@inter_args; w = @w(@ij))
        return nothing
    end
    return nothing
end

@inline function sFilter!(@self_args)::Nothing
    @inbounds if @tag(@i) == FLUID_TAG
        SPH.Library.sKernelFilter!(@self_args; w0 = SPH.Kernel.value0(@h(@i), sph_kernel))
        SPH.Library.sVolume!(@self_args)
        @inbounds @p(@i) = eos(@rho(@i))
        return nothing
    end
    return nothing
end

# * ===================== simulation function definition ===================== * #

function main(step = :first)
    appendix = DataIO.appendix()
    DataIO.mkdir(writer)
    DataIO.load!(writer, hps; appendix = appendix, step = step)
    t = parallel(appendix["TimeValue"])
    step = parallel(appendix["TMSTEP"])
    write_step = parallel(appendix["WriteStep"])
    percentage = t * total_time_inv
    Class.asyncto!(dps, hps)
    Algorithm.search!(
        dps,
        domain,
        ns,
        n_threads = n_threads,
        action! = sComputeKernelAndiContinuity!,
        criterion = Algorithm.symmetryCriterion,
    )
    progress = ProgressMeter.ProgressThresh(0.0; desc = "Task Left:", dt = 0.1, color = color, showspeed = true)
    while t < total_time
        t += dt
        step += 1
        Algorithm.selfaction!(dps, sContinuity!; n_threads = n_threads)
        Algorithm.interaction!(dps, iMomentum!; n_threads = n_threads)
        Algorithm.selfaction!(dps, sMomentum!; n_threads = n_threads)
        Algorithm.search!(
            dps,
            domain,
            ns,
            n_threads = n_threads,
            action! = sComputeKernelAndiContinuity!,
            criterion = Algorithm.symmetryCriterion,
        )
        if step % filter_interval == 0
            Algorithm.interaction!(dps, iFilter!; n_threads = n_threads)
            Algorithm.selfaction!(dps, sFilter!; n_threads = n_threads)
        end
        if step % output_interval == 0
            write_step += 1
            appendix["TimeValue"] = t
            appendix["TMSTEP"] = step
            appendix["WriteStep"] = write_step
            DataIO.wait!(writer)
            Class.asyncto!(hps, dps)
            DataIO.save!(writer, hps, appendix)
        end
        percentage = t * total_time_inv
        update!(
            progress,
            1 - percentage;
            showvalues = [
                ("Backend", backend),
                ("TimeValue", t),
                ("TMSTEP", step),
                ("WriteStep", write_step),
                ("Percentage %", percentage * 100),
            ],
            valuecolor = color,
        )
    end
    DataIO.wait!(writer)
    finish!(progress)
    VTK.VTP.generate(writer, [:Mass, :Density, :VelocityVec, :Pressure, :Tag]; names = ["fluid", "wall"], n_threads = 4)
    return nothing
end
