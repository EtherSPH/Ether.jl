#=
  @ author: bcynuaa <bcynuaa@163.com>
  @ date: 2025/05/20 16:13:13
  @ license: MIT
  @ language: Julia
  @ declaration: `Ether.jl` A particle-based simulation framework running on both cpu and gpu.
  @ description:
 =#

using Ether
using CairoMakie

result_path = joinpath(@__DIR__, "../../../result/sph/cruchaga/3d_extrapolation/")
writer = DataIO.Writer(path_ = result_path);
DataIO.readRaw!(writer)

appendix = DataIO.appendix();
ps = DataIO.load(writer; appendix = appendix, step = 70)
tag = ps[:Tag]
fpos = ps[:PositionVec][tag .== 1, :]
wpos = ps[:PositionVec][tag .== 2, :]
vels = ps[:VelocityVec][tag .== 1, :]
vel = sqrt.(vels[:, 1] .^ 2 .+ vels[:, 2] .^ 2 .+ vels[:, 3] .^ 2);

with_theme(theme_latexfonts()) do
    t = appendix["TimeValue"]
    t = round(t, digits = 6) |> string
    fig = Figure(size = (800, 800), fontsize = 18)
    axes = Axis3(
        fig[1, 1],
        aspect = :data,
        title = L"$\sqrt{u^2 + v^2} = |\vec{u}|$ at Time %$t s",
        xlabel = L"x",
        ylabel = L"y",
        zlabel = L"z",
    )
    scatter!(
        axes,
        wpos,
        color = :gray,
        markersize = 5,
        marker = :rect,
        alpha = 0.2,
        strokecolor = :black,
        strokewidth = 0.1,
        colorrange = [0.0, 1.6],
    )
    s = scatter!(
        axes,
        fpos,
        color = vel,
        colormap = :roma,
        markersize = 5,
        marker = :circle,
        alpha = 0.8,
        colorrange = [0.0, 1.6],
    )
    Colorbar(fig[1, 2], s, height = 300, width = 10, label = "Velocity", ticklabelsize = 16, labelpadding = 5)
    fig
end
