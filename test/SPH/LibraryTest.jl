#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/13 16:02:36
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Library" begin
    domain = Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
    active_pair = [1 => 1, 1 => 2, 2 => 1]
    periodic_boundary = Class.PeriodicBoundary2D{false, false}
    neighbour_system = Class.NeighbourSystem(periodic_boundary, parallel, domain, active_pair)
    dim = 2
    neighbour_count = 50
    n_particles = 9
    int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
    float_named_tuple = (
        PositionVec = dim,
        nRVec = dim * neighbour_count,
        nR = neighbour_count,
        VelocityVec = dim,
        dVelocityVec = dim,
        AccelerationVec = dim,
        Mass = 1,
        Density = 1,
        dDensity = 1,
        Volume = 1,
        Pressure = 1,
        Gap = 1,
        H = 1,
        SumWeight = 1,
        SumWeightedDensity = 1,
        SumWeightedPressure = 1,
        nW = neighbour_count,
        nDW = neighbour_count,
        nHInv = neighbour_count,
    )
    particle_system =
        Class.ParticleSystem(Environment.Dimension2D, parallel, n_particles, int_named_tuple, float_named_tuple;)
    cpu_particle_system = Class.mirror(particle_system)
    start_x = Class.get_first_x(domain)
    start_y = Class.get_first_y(domain)
    last_x = Class.get_last_x(domain)
    last_y = Class.get_last_y(domain)
    gap_x = Class.get_gap_x(domain)
    gap_y = Class.get_gap_y(domain)
    gap = Class.get_gap(domain)
    xy = zeros(FT, 9, 2)
    err = FT(1e-3)
    xy[1, :] .= [start_x + err, start_y + err]
    xy[2, :] .= [start_x + gap / 2, start_y + err]
    xy[3, :] .= [start_x + gap * 3 / 2 - err, start_y + err]
    xy[4, :] .= [start_x + err, start_y + gap / 2]
    xy[5, :] .= [start_x + gap / 2, start_y + gap / 2]
    xy[6, :] .= [start_x + gap * 3 / 2 - err, start_y + gap / 2]
    xy[7, :] .= [start_x + err, start_y + gap - err]
    xy[8, :] .= [start_x + gap / 2, start_y + gap - err]
    xy[9, :] .= [start_x + gap * 3 / 2 - err, start_y + gap - err]
    for i in 1:9
        cpu_particle_system.base_.float_[i, cpu_particle_system.named_index_.index_.PositionVec + 0] = xy[i, 1]
        cpu_particle_system.base_.float_[i, cpu_particle_system.named_index_.index_.PositionVec + 1] = xy[i, 2]
        cpu_particle_system.base_.is_alive_[i] = 1
        cpu_particle_system.base_.int_[i, cpu_particle_system.named_index_.index_.Tag] = 1
        cpu_particle_system.base_.float_[i, cpu_particle_system.named_index_.index_.Gap] = FT(0.1)
        cpu_particle_system.base_.float_[i, cpu_particle_system.named_index_.index_.H] = FT(0.15)
    end
    Class.to!(particle_system, cpu_particle_system)
    hinv = 1 / (FT(1.5) * gap)
    param = (c0 = FT(340.0), gamma = IT(7), mu = FT(1e-3), h0 = FT(1.5 * gap))
    some_array = CT(FT[1.0, 2.0])
    Algorithm.search!(particle_system, domain, neighbour_system; parameter = param)
    kernel = SPH.Kernel.CubicSpline{IT, FT, dim}()
    @inline function inter!(@inter_args)::Nothing
        w = SPH.Library.iValue(@inter_args, kernel)
        dw = SPH.Library.iGradient(@inter_args, kernel)
        SPH.Library.iValue!(@inter_args, kernel)
        SPH.Library.iGradient!(@inter_args, kernel)
        SPH.Library.iValueGradient!(@inter_args, kernel)
        SPH.Library.iClassicContinuity!(@inter_args; dw = dw)
        SPH.Library.iBalancedContinuity!(@inter_args; dw = dw)
        coeff = SPH.Library.iPressureCorrection(@inter_args, kernel; hinv = hinv, w = w)
        SPH.Library.iClassicPressure!(@inter_args; dw = dw, coefficient = coeff)
        SPH.Library.iBalancedPressure!(@inter_args; dw = dw, coefficient = coeff)
        SPH.Library.iDensityWeightedPressure!(@inter_args; dw = dw, coefficient = coeff)
        SPH.Library.iExtrapolatePressure!(@inter_args; w = w, p0 = 0.0, gx = 0.0, gy = -9.8)
        SPH.Library.iClassicViscosity!(@inter_args; dw = dw, mu = 1e-3)
        SPH.Library.IArtificialViscosity!(@inter_args; dw = dw, alpha = 0.1, beta = 0.1, c = PARAMETER.c0)
        SPH.Library.iKernelFilter!(@inter_args; w = w)
        return nothing
    end
    @inline function self!(@self_args)::Nothing
        w0 = SPH.Kernel.value0(PARAMETER.h0, kernel)
        SPH.Library.sVolume!(@self_args)
        SPH.Library.sGravity!(@self_args; gx = 0.0, gy = -9.8)
        SPH.Library.sContinuity!(@self_args; dt = 0.1)
        SPH.Library.sAcceleration!(@self_args)
        SPH.Library.sAccelerate!(@self_args; dt = 0.1)
        SPH.Library.sMove!(@self_args; dt = 0.1)
        SPH.Library.sAccelerateMove!(@self_args; dt = 0.1)
        SPH.Library.sExtrapolatePressure!(@self_args; p0 = 0.0)
        SPH.Library.sKernelFilter!(@self_args; w0 = w0)
        SPH.Library.sApplyPeriodic!(@self_args, domain, Class.NonePeriodicBoundary)
        SPH.Library.sApplyPeriodic!(@self_args, domain, Class.PeriodicBoundary2D{false, false})
        SPH.Library.sApplyPeriodic!(@self_args, domain, Class.PeriodicBoundary2D{true, false})
        SPH.Library.sApplyPeriodic!(@self_args, domain, Class.PeriodicBoundary2D{false, true})
        SPH.Library.sApplyPeriodic!(@self_args, domain, Class.PeriodicBoundary2D{true, true})
        return nothing
    end
    Algorithm.interaction!(particle_system, inter!; parameter = param)
    Algorithm.selfaction!(particle_system, self!; parameter = param)
end
