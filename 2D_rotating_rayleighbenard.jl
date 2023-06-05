using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
# using GLMakie
using CairoMakie
using Oceananigans.Grids: halo_size
using ArgParse
using RescaledGeophysicalFluids

function parse_commandline()
  s = ArgParseSettings()

  @add_arg_table! s begin
    "--rayleigh_number"
      help = "Rayleigh Number"
      arg_type = Float64
      default = 2000.
    "--taylor_number"
      help = "Taylor Number"
      arg_type = Float64
      default = 0.
    "--prandtl_number"
      help = "Prandtl Number"
      arg_type = Float64
      default = 1.
    "--aspect_ratio"
      help = "aspect ratio alpha = H / L, H = 1"
      arg_type = Float64
      default = 0.125
    "--dt"
      help = "Initial timestep to take (seconds)"
      arg_type = Float64
      default = 1e-6
    "--max_dt"
      help = "Maximum timestep (seconds)"
      arg_type = Float64
      default = 1e-4
    "--stop_time"
      help = "Stop time of simulation (seconds)"
      arg_type = Float64
      default = 3.
    "--time_interval"
      help = "Time interval of output writer (seconds)"
      arg_type = Float64
      default = 1e-2
    "--fps"
      help = "Frames per second of animation"
      arg_type = Float64
      default = 20.
    "--Nz"
      help = "Number of grid points in z-direction"
      arg_type = Int64
      default = 32
    "--uv_top"
      help = "Type of boundary condition for velocities at the top"
      arg_type = String
      default = "f"
    "--uv_bot"
      help = "Type of boundary condition for velocities at the bottom"
      arg_type = String
      default = "f"
    "--b_top"
      help = "Type of boundary condition for buoyancy at the top"
      arg_type = String
      default = "v"
    "--b_bot"
      help = "Type of boundary condition for buoyancy at the bottom"
      arg_type = String
      default = "v"
  end
  return parse_args(s)
end

args = parse_commandline()

const aspect_ratio = args["aspect_ratio"]

const Lz = 1meter    # depth [m]
const Lx = Lz / aspect_ratio # north-south extent [m]

const Nz = args["Nz"]
const Nx = Int(round(Nz / aspect_ratio))

const Pr = args["prandtl_number"]
const ν = 1
const κ = ν / Pr

const Ra = args["rayleigh_number"]
const S = Ra * ν * κ / Lz ^ 4

const Ta = args["taylor_number"]
const f = √(Ta * ν ^ 2 / Lz ^ 4)

uv_bc_top_type = args["uv_top"]
uv_bc_bot_type = args["uv_bot"]

b_bc_top_type = args["b_top"]
b_bc_bot_type = args["b_bot"]

Δt = args["dt"]
max_Δt = args["max_dt"]
stop_time = args["stop_time"]
time_interval = args["time_interval"]
fps = args["fps"]

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
simulation = Simulation(model, Δt=Δt, stop_time=stop_time)

# simulation.stop_iteration = 30000

wizard = TimeStepWizard(max_change=1.05, max_Δt=max_Δt, cfl=0.6)
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
                                                          schedule = TimeInterval(time_interval),
                                                          # schedule = IterationInterval(10),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, U, V, W, UW, VW, WW, WB, UV, VV, VB, UU, UB),
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          # schedule = IterationInterval(10),
                                                          schedule = TimeInterval(time_interval),
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

bs_k = interior(b_data[end], :, 1, Int(floor(length(zb) / 2)))

kc, kc_neighbourhood = calculate_critical_k(bs_k, xb)

@info "Critical k = $(kc) in the neighbourhood of $(kc_neighbourhood)"

##
fig = Figure(resolution=(1200, 1500))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

axb = Axis(fig[1, 1:2], title="b, k = $(round(kc, digits=2)), neighbouring ks = $(round.(kc_neighbourhood, digits=2))", xlabel="x", ylabel="z")
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

record(fig, "$(FILE_DIR)/$(FILE_DIR).mp4", 1:Nt, framerate=20) do nn
    n[] = nn
end
##