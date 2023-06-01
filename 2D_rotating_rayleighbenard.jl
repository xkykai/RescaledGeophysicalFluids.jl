using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
# using GLMakie
using CairoMakie
using Oceananigans.Grids: halo_size

const aspect_ratio = 0.125

const Lz = 1meter    # depth [m]
const Lx = Lz / aspect_ratio # north-south extent [m]

const Nz = 64
const Nx = Int(Nz / aspect_ratio)

const Pr = 1
const ν = 1
const κ = ν / Pr

const Ra = 5000
const S = Ra * ν * κ / Lz ^ 4

const Ta = 1000
const f = √(Ta * ν ^ 2 / Lz ^ 4)

uv_bc_top_type = "f"
uv_bc_bot_type = "f"

b_bc_top_type = "v"
b_bc_bot_type = "v"

FILE_DIR = "Data/2D_no_wind_uv_t$(uv_bc_top_type)b$(uv_bc_bot_type)_b_t$(b_bc_top_type)b$(b_bc_bot_type)_Ra_$(Ra)_Ta_$(Ta)_alpha_$(aspect_ratio)_Nz_$(Nz)"
mkpath(FILE_DIR)

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Nz),
                       halo = (4, 4),
                       x = (0, Lx),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

if uv_bc_top_type == "f"
  uv_bc_top = GradientBoundaryCondition(0)
else
  uv_bc_top = ValueBoundaryCondition(0)
end

if uv_bc_bot_type == "f"
  uv_bc_bot = GradientBoundaryCondition(0)
else
  uv_bc_bot = ValueBoundaryCondition(0)
end

if b_bc_top_type == "f"
  b_bc_top = GradientBoundaryCondition(-S)
else
  b_bc_top = ValueBoundaryCondition(-S * Lz)
end

if b_bc_bot_type == "f"
  b_bc_bot = GradientBoundaryCondition(0)
else
  b_bc_bot = ValueBoundaryCondition(0)
end

uv_bcs = FieldBoundaryConditions(top=uv_bc_top, bottom=uv_bc_bot)

b_bcs = FieldBoundaryConditions(top=b_bc_top, bottom=b_bc_bot)

b_initial(x, y, z) = -rand() * Ra / 100000

model = NonhydrostaticModel(; 
            grid = grid,
            closure = ScalarDiffusivity(ν=ν, κ=κ),
            coriolis = FPlane(f=f), # note that f is positive
            buoyancy = BuoyancyTracer(),
            tracers = :b,
            timestepper = :RungeKutta3,
            advection = WENO(),
            boundary_conditions=(; u=uv_bcs, v=uv_bcs, b=b_bcs)
            )

set!(model, b=b_initial)

# simulation = Simulation(model, Δt=2e-6second, stop_iteration=2000)
simulation = Simulation(model, Δt=1e-6second, stop_time=3seconds)

# simulation.stop_iteration = 30000

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
# simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(10))

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

ζ = Field(∂x(v))
∂z_b = Field(∂z(b))

PV = Field(ζ - f * ∂z_b / S)
compute!(PV)

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

field_outputs = merge(model.velocities, model.tracers, (; PV))

simulation.output_writers[:jld2] = JLD2OutputWriter(model, field_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_fields.jld2",
                                                          schedule = TimeInterval(5e-3),
                                                          # schedule = IterationInterval(10),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, U, V, W, UW, VW, WW, WB, UV, VV, VB, UU, UB),
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          # schedule = IterationInterval(10),
                                                          schedule = TimeInterval(5e-3),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:averages] = JLD2OutputWriter(model, (; B, U, V, W, UW, VW, WW, WB, UV, VV, VB, UU, UB),
                                                        filename = "$(FILE_DIR)/averaged_timeseries.jld2",
                                                        schedule = AveragedTimeInterval(1seconds, window=1second),
                                                        with_halos = true,
                                                        init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(1seconds), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration10000.jld2")
run!(simulation)

metadata = FieldDataset("$(FILE_DIR)/instantaneous_fields.jld2").metadata

b_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b")
PV_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "PV")

B_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "B")
WB_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "WB")

Nu_data = WB_data ./ (κ * S)

Nt = length(b_data.times)

xb, yb, zb = nodes(b_data)
xPV, yPV, zPV = nodes(PV_data)

xB, yB, zB = nodes(B_data)
xNu, yNu, zNu = nodes(WB_data)

##
fig = Figure(resolution=(1200, 1500))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

axb = Axis(fig[1, 1:2], title="b", xlabel="x", ylabel="z")
axPV = Axis(fig[2, 1:2], title="PV", xlabel="x", ylabel="z")

axB = Axis(fig[3, 1], title="<b>", xlabel="<b>", ylabel="z")
axNu = Axis(fig[3, 2], title="Nu", xlabel="Nu", ylabel="z")

bn = @lift interior(b_data[$n], :, 1, :)
PVn = @lift interior(PV_data[$n], :, 1, :)

Bn = @lift interior(B_data[$n], 1, 1, :)
Nun = @lift Nu_data[1, 1, :, $n]
time_str = @lift "Ra = $(Ra), Ta = $(Ta), Pr = $(Pr), Time = $(round(b_data.times[$n], digits=2)), Maximum |PV| = $(round(maximum(abs, interior(PV_data[$n])), digits=1))"

title = Label(fig[-1, 1:2], time_str, font=:bold)

blim = (minimum(b_data), maximum(b_data))
PVlim = (-maximum(abs, interior(PV_data[end], :, 1, :)), maximum(abs, interior(PV_data[end], :, 1, :)))

Blim = (minimum(B_data), maximum(B_data))
Nulim = (minimum(Nu_data), maximum(Nu_data))

heatmap!(axb, xb, zb, bn, colormap=Reverse(:RdBu_10), colorrange=blim)
heatmap!(axPV, xPV, zPV, PVn, colormap=Reverse(:RdBu_10), colorrange=PVlim)
lines!(axB, Bn, zB)
lines!(axNu, Nun, zNu)

xlims!(axB, Blim)
xlims!(axNu, Nulim)

record(fig, "$(FILE_DIR)/rayleighbenard_convection.mp4", 1:Nt, framerate=30) do nn
    n[] = nn
end
##