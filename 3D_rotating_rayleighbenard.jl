using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
# using GLMakie
using CairoMakie
using Oceananigans.Grids: halo_size

const aspect_ratio = 0.25

const Lz = 1meter    # depth [m]
const Lx = Lz / aspect_ratio # north-south extent [m]
const Ly = Lx

const Nz = 64
const Nx = Int(Nz / aspect_ratio)
const Ny = Nx

const Pr = 1
const ν = 1
const κ = ν / Pr

const Ra = 2151.3411993087107
const S = Ra * ν * κ / Lz ^ 4

const Ta = 1000
const f = √(Ta * ν ^ 2 / Lz ^ 4)

FILE_DIR = "Data/3D_no_wind_Ra_$(Ra)_Ta_$(Ta)_alpha_$(aspect_ratio)_Nz_$(Nz)"
# FILE_DIR = "Data/3D_no_wind_Ra_2163.3411993087107_Ta_1000_alpha_1_Nz_64"
mkpath(FILE_DIR)

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Ny, Nz),
                       halo = (4, 4, 4),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = (0, Lz),
                       topology = (Periodic, Periodic, Bounded))

b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(-S * Lz),
                                bottom = ValueBoundaryCondition(0))

b_initial(x, y, z) = -rand() * Ra / 10000

model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f), # note that f is positive
            buoyancy = BuoyancyTracer(),
            tracers = :b,
            timestepper = :RungeKutta3,
            advection = WENO(),
            boundary_conditions=(; b=b_bcs)
            )

set!(model, b=b_initial)

# simulation = Simulation(model, Δt=4e-6second, stop_iteration=100000)
simulation = Simulation(model, Δt=1e-6second, stop_time=10seconds)

wizard = TimeStepWizard(max_change=1.05, max_Δt=5e-6, cfl=0.6)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

wall_clock = [time_ns()]

function print_progress(sim)
    @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.velocities.u),
            maximum(abs, sim.model.velocities.v),
            maximum(abs, sim.model.velocities.w),
            maximum(abs, sim.model.tracers.b),
            prettytime(sim.Δt))

    wall_clock[1] = time_ns()

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1000))
# simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

function init_save_some_metadata!(file, model)
    file["metadata/author"] = "Xin Kai Lee"
    file["metadata/parameters/coriolis_parameter"] = f
    file["metadata/parameters/density"] = 1027
    file["metadata/parameters/rayleigh_number"] = Ra
    file["metadata/parameters/prandtl_number"] = Pr
    file["metadata/parameters/taylor_number"] = Ta
    file["metadata/parameters/aspect_ratio"] = aspect_ratio
    return nothing
end

b = model.tracers.b
u, v, w = model.velocities

B = Average(b, dims=(1, 2))
U = Average(u, dims=(1, 2))
V = Average(v, dims=(1, 2))
W = Average(w, dims=(1, 2))

UW = Average(w * u, dims=(1, 2))
VW = Average(w * v, dims=(1, 2))
WW = Average(w * w, dims=(1, 2))
WB = Average(w * b, dims=(1, 2))

UV = Average(v * u, dims=(1, 2))
VV = Average(v * v, dims=(1, 2))
VB = Average(v * b, dims=(1, 2))

UU = Average(u * u, dims=(1, 2))
UB = Average(u * b, dims=(1, 2))

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(2e-3),
                                                          # schedule = IterationInterval(10),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, U, V, W, UW, VW, WW, WB, UV, VV, VB, UU, UB),
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          # schedule = IterationInterval(10),
                                                          schedule = TimeInterval(2e-3),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; B, U, V, W, UW, VW, WW, WB, UV, VV, VB, UU, UB),
                                                        filename = "$(FILE_DIR)/averaged_timeseries.jld2",
                                                        schedule = AveragedTimeInterval(1seconds, window=1second),
                                                        with_halos = true,
                                                        init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1seconds), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration400195.jld2")
run!(simulation)

b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b", backend=OnDisk())
B_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "B")
WB_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "WB")

Nu_data = WB_data ./ (κ * S)

Nt = length(b_data.times)

xC = b_data.grid.xᶜᵃᵃ[1:Nx]
yC = b_data.grid.xᶜᵃᵃ[1:Ny]
zC = b_data.grid.zᵃᵃᶜ[1:Nz]

zF = WB_data.grid.zᵃᵃᶠ[1:Nz+1]
##
fig = Figure(resolution=(1000, 1250))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

axb = Axis3(fig[1:2, 1:2], title="b", xlabel="x", ylabel="y", zlabel="z", viewmode=:fitzoom)
axB = Axis(fig[3, 1], title="<b>", xlabel="<b>", ylabel="z")
axNu = Axis(fig[3, 2], title="Nu", xlabel="Nu", ylabel="z")

xs_xy = xC
ys_xy = yC
zs_xy = [zC[Nz] for x in xs_xy, y in ys_xy]

ys_yz = yC
xs_yz = range(xC[1], stop=xC[1], length=length(zC))
zs_yz = zeros(length(xs_yz), length(ys_yz))
for j in axes(zs_yz, 2)
  zs_yz[:, j] .= zC
end

xs_xz = xC
ys_xz = range(yC[1], stop=yC[1], length=length(zC))
zs_xz = zeros(length(xs_xz), length(ys_xz))
for i in axes(zs_xz, 1)
  zs_xz[i, :] .= zC
end

blim = (minimum(b_data), maximum(b_data))

colormap = Reverse(:RdBu_10)
color_range = blim

Blim = (minimum(B_data), maximum(B_data))
Nulim = (minimum(Nu_data), maximum(Nu_data))

bn_xy = @lift interior(b_data[$n], :, :, Nz)
bn_yz = @lift transpose(interior(b_data[$n], 1, :, :))
bn_xz = @lift interior(b_data[$n], :, 1, :)
time_str = @lift "Ra = $(Ra), Ta = $(Ta), Pr = $(Pr), Time = $(round(b_data.times[$n], digits=2))"
title = Label(fig[-1, :], time_str, font=:bold, tellwidth=false)

b_xy_surface = surface!(axb, xs_xy, ys_xy, zs_xy, color=bn_xy, colormap=colormap, colorrange = color_range)
b_yz_surface = surface!(axb, xs_yz, ys_yz, zs_yz, color=bn_yz, colormap=colormap, colorrange = color_range)
b_xz_surface = surface!(axb, xs_xz, ys_xz, zs_xz, color=bn_xz, colormap=colormap, colorrange = color_range)

Bn = @lift interior(B_data[$n], 1, 1, :)
Nun = @lift Nu_data[1, 1, :, $n]

lines!(axB, Bn, zC)
lines!(axNu, Nun, zF)

xlims!(axB, Blim)
xlims!(axNu, Nulim)

trim!(fig.layout)
# display(fig)

record(fig, "$(FILE_DIR)/rayleighbenard_convection.mp4", 1:Nt, framerate=30) do nn
    n[] = nn
end
##