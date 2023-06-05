module RescaledGeophysicalFluids

export calculate_critical_k,
       FluxBoundaryCondition, ValueBoundaryCondition, 
       evaluate_m_expr, compute_coefficients, build_bc_matrix, find_logdet_A, find_critical_Ra, find_critical_Ra_k, find_plot_critical_Ra_k_A, find_zero_eigenvalue
       
using SymPy
using LinearAlgebra
using Roots
using CairoMakie
using Optim
using FFTW
using Oceananigans

include("wavenumber_analysis.jl")
include("linear_stability_analysis.jl")

end # module RescaledGeophysicalFluids
