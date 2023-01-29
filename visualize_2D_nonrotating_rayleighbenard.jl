using Oceananigans
using Oceananigans.Grids
using JLD2
using FileIO
using Printf
using GLMakie

FILE_DIR = "Data/2D_no_wind_Ra_1.0e6_Ta_0_alpha_0.25"

b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_b.jld2", "b")

u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_velocities.jld2", "u")
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_velocities.jld2", "w")

Nt = length(b_data.times)

xs = xnodes(Center, b_data.grid)
zs = znodes(Center, b_data.grid)

##
fig = Figure(resolution=(3000, 1000))

slider = Slider(fig[2, 1:3], range=1:Nt, startvalue=1)
n = slider.value

# suptitle = LText(fig[0, 1:3], "thing")

n = Observable(1)
axb = Axis(fig[1, 1], title="b", xlabel="x/H", ylabel="z/H")
axu = Axis(fig[1, 2], title="u", xlabel="x/H", ylabel="z/H")
axw = Axis(fig[1, 3], title="w", xlabel="x/H", ylabel="z/H")

bn = @lift interior(b_data[$n], :, 1, :)
un = @lift interior(u_data[$n], :, 1, :)
wn = @lift interior(w_data[$n], :, 1, :)

blim = maximum(abs, b_data)
ulim = maximum(abs, u_data)
wlim = maximum(abs, w_data)

heatmap!(axb, xs, zs, bn, colormap=Reverse(:RdBu_10), colorrange=(-blim, blim))
heatmap!(axu, xs, zs, un, colormap=Reverse(:RdBu_10), colorrange=(-ulim, ulim))
heatmap!(axw, xs, zs, wn, colormap=Reverse(:RdBu_10), colorrange=(-wlim, wlim))

display(fig)
##

record(fig, "$(FILE_DIR)/rayleighbenard_convection.mp4", 1:Nt, framerate=30) do nn
    n[] = nn
end
