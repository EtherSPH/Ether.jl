#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/19 20:11:04
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description: # ! KernelAbstractions seems not to have supported such kind of args passing yet
 =#

include("../../head/cpu.jl")

function f!(I, a; dt::Float32 = 1.0f0)
    @inbounds a[I] += a[I] * dt
end

@kernel function device_apply_action!(a, action!; parameters...)
    I = @index(Global)
    action!(I, a; parameters...)
end

function host_apply_action!(a, action!; parameters...)
    device_apply_action!(Backend, 8)(a, action!; parameters..., ndrange = (length(a),))
    KernelAbstractions.synchronize(Backend)
end

a = ones(FT, 10) |> CT
@info "a = $a"
host_apply_action!(a, f!)
@info "`host_apply_action!(a, f!)` is allowed!"
@info "a = $a"
host_apply_action!(a, f!; dt = 0.2f0)
@info "`host_apply_action!(a, f!; dt = 0.2f0)` is not allowed"
@info "a = $a"
