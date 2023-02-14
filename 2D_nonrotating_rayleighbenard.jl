using Oceananigans
using Oceananigans.Units
using JLD2
using FileIO
using Printf
# using GLMakie
# using CairoMakie

const aspect_ratio = 0.25

const Lz = 1meter    # depth [m]
const Lx = Lz / aspect_ratio # north-south extent [m]

const Nx = 256
const Nz = 256

const Pr = 1
const ν = 1
const κ = ν / Pr

const Ra = 1e6
const S = Ra * ν * κ / Lz ^ 4

const Ta = 0
const f = √(Ta * ν ^ 2 / Lz ^ 4)

FILE_DIR = "Data/2D_no_wind_Ra_$(Ra)_Ta_$(Ta)_alpha_$(aspect_ratio)"
mkpath(FILE_DIR)

grid = RectilinearGrid(GPU(), Float64,
                       size = (Nx, Nz),
                       halo = (4, 4),
                       x = (0, Lx),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(-S * Lz),
                                bottom = ValueBoundaryCondition(0))

b_initial(x, y, z) = -rand() * Ra / 100000

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

# simulation = Simulation(model, Δt=1e-6second, stop_iteration=20)
simulation = Simulation(model, Δt=1e-6second, stop_time=20seconds)

# simulation.stop_iteration = 30000

wizard = TimeStepWizard(max_change=1.05, max_Δt=1e-5)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(50))

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
# simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

function init_save_some_metadata!(file, model)
    file["author"] = "Xin Kai Lee"
    file["parameters/coriolis_parameter"] = f
    file["parameters/density"] = 1027
    file["parameters/rayleigh_number"] = Ra
    file["parameters/prandtl_number"] = Pr
    file["parameters/taylor_number"] = Ta
    file["parameters/aspect_ratio"] = aspect_ratio
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

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=TimeInterval(10seconds), prefix="$(FILE_DIR)/model_checkpoint")

# run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration10000.jld2")
run!(simulation)