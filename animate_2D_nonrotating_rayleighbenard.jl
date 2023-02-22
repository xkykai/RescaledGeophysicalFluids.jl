using GLMakie
using Oceananigans
using Oceananigans.Grids: halo_size

FILE_DIR = "./Data/2D_no_wind_Ra_1.0e6_Ta_0_alpha_0.25"

b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b")

B_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "B")
WB_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "WB")

Nu_data = WB_data ./ (κ * S)

# u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u")
# v_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "v")
# w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
interior(WB_data, 1, 1, :, :) ./ (κ * S)

Lz = b_data.grid.Lz
Pr = 1
ν = 1
κ = ν / Pr

Ra = 1e6
S = Ra * ν * κ / Lz ^ 4
ΔB = S * Lz

Ta = 0
f = √(Ta * ν ^ 2 / Lz ^ 4)

Nt = length(b_data.times)
Nx = b_data.grid.Nx
Nz = b_data.grid.Nz

xC = B_data.grid.xᶜᵃᵃ[1:Nx]
zC = B_data.grid.zᵃᵃᶜ[1:Nz]
zF = WB_data.grid.zᵃᵃᶠ[1:Nz+1]

fig = Figure(resolution=(1500, 1500))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

axb = Axis(fig[1, 1:2], title="b", xlabel="x", ylabel="z")
axB = Axis(fig[2, 1], title="<b>", xlabel="<b>", ylabel="z")
axNu = Axis(fig[2, 2], title="Nu", xlabel="Nu", ylabel="z")

bn = @lift interior(b_data[$n], :, 1, :)
Bn = @lift interior(B_data[$n], 1, 1, :)
Nun = @lift Nu_data[1, 1, :, $n]
time_str = @lift "Time = $(round(b_data.times[$n], digits=2))"

title = Label(fig[-1, 1:2], time_str, font=:bold)

blim = (-1*maximum(abs, b_data), maximum(abs, b_data))
Blim = (minimum(B_data), maximum(B_data))
Nulim = (minimum(Nu_data), maximum(Nu_data))

heatmap!(axb, xC, zC, bn, colormap=Reverse(:RdBu_10), colorrange=blim)
lines!(axB, Bn, zC)
lines!(axNu, Nun, zF)

xlims!(axB, Blim)
xlims!(axNu, Nulim)

display(fig)

record(fig, "$(FILE_DIR)/rayleighbenard_convection.mp4", 1:300, framerate=10) do nn
    n[] = nn
end
