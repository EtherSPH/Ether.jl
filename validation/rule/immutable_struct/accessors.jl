#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 18:08:40
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # * accepted performance drop
 =#

using Accessors

@kwdef struct Simulation
    t::Float64 = 0.0
    dt::Float64 = 0.01
    step::Int32 = 0
end

@inline function step!(sim::Simulation)
    @reset sim.step += 1
    @reset sim.t += sim.dt
end

@inline function step(sim::Simulation)
    @reset sim.step += 1
    @reset sim.t += sim.dt
    return sim
end

sim = Simulation()
@info "initial: $sim"
step!(sim)
@info "after step!(sim): $sim" # can't change the field of sim
sim = step(sim)
@info "after step(sim): $sim" # can change the field of sim

const N = 10^8
@info "Benchmarking step(sim) for $N times requires:"
@time begin
    for _ in 1:(10 ^ 8)
        global sim = step(sim)
    end
end
