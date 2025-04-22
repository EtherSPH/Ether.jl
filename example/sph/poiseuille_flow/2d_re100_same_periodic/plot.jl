#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/04/22 21:59:30
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Ether
using CairoMakie

result_path = joinpath(@__DIR__, "../../../result/sph/poiseuille_flow/2d_re100_same_periodic/")
writer = DataIO.Writer(result_path);
DataIO.readRaw!(writer)

appendix = DataIO.appendix();
ps = DataIO.load(writer; appendix = appendix, step = :last);
tag = ps[:Tag]
x = ps[:PositionVec][:, 1]
y = ps[:PositionVec][:, 2]
u = ps[:VelocityVec][:, 1]
v = ps[:VelocityVec][:, 2]
vel = sqrt.(u .^ 2 .+ v .^ 2)
vel[tag .== 2] .= NaN;

with_theme(theme_latexfonts()) do
    t = appendix["TimeValue"]
    t = round(t, digits = 6) |> string
    fig = Figure(size = (800, 300), fontsize = 18)
    axes = Axis(
        fig[1, 1],
        aspect = DataAspect(),
        title = L"$\sqrt{u^2 + v^2} = |\vec{u}|$ at Time %$t s",
        xlabel = L"x",
        ylabel = L"y",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
    )
    s = scatter!(axes, x, y, color = vel, colormap = :roma, markersize = 6, nan_color = :gray)
    Colorbar(
        fig[1, 2],
        s,
        height = 300,
        width = 10,
        label = "Velocity",
        ticklabelsize = 16,
        labelpadding = 5,
        ticks = collect(0.0:0.001:0.015),
    )
    fig
end
