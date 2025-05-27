#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/27 16:34:00
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Ether
using CairoMakie

result_path = joinpath(@__DIR__, "../../../result/sph/thermal_ring/ra4.7e4_ring2.6_compulsive")
writer = DataIO.Writer(path_ = result_path);
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
T = ps[:Temperature];

with_theme(theme_latexfonts()) do
    t = appendix["TimeValue"]
    t = round(t, digits = 6) |> string
    fig = Figure(size = (1200, 400), fontsize = 18)
    axes1 = Axis(
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
    s1 = scatter!(axes1, x, y, color = vel, colormap = :roma, markersize = 3, nan_color = :gray, marker = :rect)
    Colorbar(fig[1, 2], s1, height = 300, width = 10, label = "Velocity", ticklabelsize = 16, labelpadding = 5)
    axes2 = Axis(
        fig[1, 3],
        aspect = DataAspect(),
        title = L"$T$ at Time %$t s",
        xlabel = L"x",
        ylabel = L"y",
        xgridvisible = false,
        ygridvisible = false,
        xtickalign = 1,
        ytickalign = 1,
    )
    s2 = scatter!(
        axes2,
        x,
        y,
        color = T,
        colormap = Reverse(:roma),
        markersize = 3,
        nan_color = :gray,
        marker = :rect,
    )
    Colorbar(fig[1, 4], s2, height = 300, width = 10, label = "Temperature", ticklabelsize = 16, labelpadding = 5)
    fig
end
