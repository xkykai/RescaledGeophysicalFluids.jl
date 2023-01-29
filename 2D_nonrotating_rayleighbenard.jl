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

const Nx = 200
const Nz = 200

const Pr = 1
const ν = 1
const κ = ν / Pr

const Ra = 1e6
const S = Ra * ν * κ / Lz ^ 4

const Ta = 0
const f = √(Ta * ν ^ 2 / Lz ^ 4)

FILE_DIR = "Data/2D_no_wind_Ra_$(Ra)_Ta_$(Ta)_alpha_$(aspect_ratio)"
mkpath(FILE_DIR)

grid = RectilinearGrid(CPU(), Float64,
                       size = (Nx, Nz),
                       halo = (4, 4),
                       x = (0, Lx),
                       z = (-Lz, 0),
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

simulation = Simulation(model, Δt=1e-6second, stop_iteration=10000)

simulation.stop_iteration = 30000

wizard = TimeStepWizard(max_change=1.1, max_Δt=5e-6)
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

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(1))

function init_save_some_metadata!(file, model)
    file["author"] = "Xin Kai Lee"
    file["parameters/coriolis_parameter"] = f
    file["parameters/density"] = 1027
    file["parameters/rayleigh_number"] = Ra
    file["parameters/prandtl_number"] = Pr
    file["parameters/taylor_number"] = Ta
    return nothing
end

b = model.tracers.b

simulation.output_writers[:b] = JLD2OutputWriter(model, (; b),
                                                          filename = "$(FILE_DIR)/instantaneous_b.jld2",
                                                          schedule = IterationInterval(10),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)
                                                                              

simulation.output_writers[:velocities] = JLD2OutputWriter(model, model.velocities,
                                                          filename = "$(FILE_DIR)/instantaneous_velocities.jld2",
                                                          schedule = IterationInterval(10),
                                                          with_halos = true,
                                                        #   overwrite_existing = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(1000), prefix="$(FILE_DIR)/model_checkpoint")

run!(simulation, pickup="$(FILE_DIR)/model_checkpoint_iteration10000.jld2")