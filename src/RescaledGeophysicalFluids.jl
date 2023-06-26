module RescaledGeophysicalFluids

export calculate_critical_k,
       FluxBoundaryCondition, ValueBoundaryCondition, 
       evaluate_m_expr, evaluate_m, compute_coefficients, build_bc_matrix, find_logdet_A, find_critical_Ra, find_critical_Ra_k, find_plot_critical_Ra_k_A, find_zero_eigenvalue,
       evaluate_m_expr_ω, compute_coefficients_ω, build_bc_matrix_ω, find_logdet_A_ω, find_critical_Ra_ω, find_critical_Ra_k_ω, find_plot_critical_Ra_k_A_ω
       
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
