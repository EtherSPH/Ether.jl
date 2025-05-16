#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/11 21:42:03
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

@testset "Search" begin
    @testset "Search 2D" begin
        domain = Class.Domain2D{IT, FT}(0.15, 0.1, 0.2, 0.9, 0.9)
        active_pair = [1 => 1, 1 => 2, 2 => 1]
        periodic_boundary = Class.PeriodicBoundary2D{false, false}
        neighbour_system = Class.NeighbourSystem(periodic_boundary, parallel, domain, active_pair)
        dim = 2
        neighbour_count = 50
        n_particles = 9
        int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = 1 * neighbour_count)
        float_named_tuple = (PositionVec = dim, nRVec = dim * neighbour_count, nR = neighbour_count)
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
        end
        Class.to!(particle_system, cpu_particle_system)
        param = (a = 1, b = 1.0f0)
        some_array = CT(FT[1.0, 2.0])
        Algorithm.search!(
            particle_system,
            domain,
            neighbour_system;
            n_threads = 32,
            launch = Algorithm.kDynamic,
            parameter = param,
            action! = Algorithm.defaultSelfaction!,
            criterion = Algorithm.defaultCriterion,
        )
        Algorithm.search!(
            particle_system,
            domain,
            neighbour_system;
            n_threads = 32,
            launch = Algorithm.kStatic,
            parameter = some_array,
            action! = Algorithm.defaultSelfaction!,
            criterion = Algorithm.defaultCriterion,
        )
        Class.to!(cpu_particle_system, particle_system)
        @test cpu_particle_system.base_.int_[:, cpu_particle_system.named_index_.index_.nCount] ==
              [4, 5, 3, 5, 6, 3, 4, 5, 3]
        res = [
            sort([2, 4, 5, 7]),
            sort([1, 4, 5, 8, 3]),
            sort([2, 6, 9]),
            sort([1, 2, 5, 7, 8]),
            sort([1, 2, 4, 7, 8, 6]),
            sort([5, 3, 9]),
            sort([1, 4, 5, 8]),
            sort([2, 4, 5, 7, 9]),
            sort([8, 3, 6]),
        ]
        for i in 1:9
            count = cpu_particle_system.base_.int_[i, cpu_particle_system.named_index_.index_.nCount]
            n_index = cpu_particle_system.named_index_.index_.nIndex
            @test sort(cpu_particle_system.base_.int_[i, n_index:(n_index + count - 1)]) == res[i]
        end
    end
    @testset "Search 3D" begin
        domain = Class.Domain3D{Int32, Float32}(0.15, 0, 0, 0, 0.4, 0.5, 0.7)
        ns = Class.NeighbourSystem(Class.NonePeriodicBoundary, parallel, domain, [1=>1, 1=>2, 2=>1, 2=>2])
        dimension = 3
        max_neighbour_number = 20
        int_named_tuple = (Tag = 1, IsMovable = 1, nCount = 1, nIndex = max_neighbour_number)
        float_named_tuple = (
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
        cuboid = Geometry.Cuboid(0.0, 0, 0, 0.2, 0.3, 0.4);
        n = Geometry.count(0.1, cuboid)
        ps = Class.HostParticleSystem{Int32, Float32, Environment.Dimension3D}(n, int_named_tuple, float_named_tuple)
        Class.count!(ps, n)
        positions, volumes, gaps = Geometry.create(0.1, cuboid; parallel = true)
        ps[:PositionVec] = positions
        ps[:Volume] = volumes
        ps[:Gap] = gaps
        ps[:Tag] = 1;
        dps = Class.mirror(parallel, ps)
        Algorithm.search!(dps, domain, ns);
        Class.to!(ps, dps)
        @test ps[:nCount] == IT[6, 6, 9, 9, 6, 6, 9, 9, 13, 13, 9, 9, 9, 9, 13, 13, 9, 9, 6, 6, 9, 9, 6, 6]
    end
end
