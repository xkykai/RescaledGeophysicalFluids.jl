using RescaledGeophysicalFluids
using LinearAlgebra
using CairoMakie

ms_eq = evaluate_m_expr()

Ta = 1000

α = 1
Pr = 1
ν = 1
H = 1
ρ₀ = 1000

##
# uv: top flux, bottom flux, b: top value, bottom value
bcs_type_uv_tfbf_b_tvbv = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))
                              
ks_uv_tfbf_b_tvbv = 3:0.01:4
res_Ra_k_uv_tfbf_b_tvbv, fig_uv_tfbf_b_tvbv, A_uv_tfbf_b_tvbv, A_eigen_uv_tfbf_b_tvbv, k′_uv_tfbf_b_tvbv, m′s_uv_tfbf_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tvbv, ms_eq, ks_uv_tfbf_b_tvbv, Ta, Pr; Ra_min=Ta*1.5, Ra_max=1e4)

RHS_uv_tfbf_b_tvbv = [0, 0, 0, 0, 0, 0, 0, res_Ra_k_uv_tfbf_b_tvbv.minimum]

P_uv_tfbf_b_tvbv = A_uv_tfbf_b_tvbv \ RHS_uv_tfbf_b_tvbv

coeffs_uv_tfbf_b_tvbv = compute_coefficients(k′_uv_tfbf_b_tvbv, m′s_uv_tfbf_b_tvbv; Ra=res_Ra_k_uv_tfbf_b_tvbv.minimum, Ta=Ta, Pr=Pr, ρ₀=ρ₀, H=H, ν=ν)

V_uv_tfbf_b_tvbv = [coeffs_uv_tfbf_b_tvbv[i][2] * P_uv_tfbf_b_tvbv[i] for i in eachindex(P_uv_tfbf_b_tvbv)]

zs = -H/2:0.01:H/2
xs = 0:0.01:4H

vs_uv_tfbf_b_tvbv = zeros(length(xs), length(zs))
for i in eachindex(xs), j in eachindex(zs)
    vs_uv_tfbf_b_tvbv[i, j] = real(sum([V_uv_tfbf_b_tvbv[i′] * exp(m′s_uv_tfbf_b_tvbv[i′] * zs[j]) for i′ in eachindex(V_uv_tfbf_b_tvbv)]) * exp(im * k′_uv_tfbf_b_tvbv * xs[i]))
end

f = √(Ta * ν ^ 2 / H ^ 4)

v_gs_uv_tfbf_b_tvbv = zeros(length(xs), length(zs))
for i in eachindex(xs), j in eachindex(zs)
    v_gs_uv_tfbf_b_tvbv[i, j] = real(im * k′_uv_tfbf_b_tvbv * exp(im * k′_uv_tfbf_b_tvbv * xs[i]) * 
                                   sum([P_uv_tfbf_b_tvbv[i′] * exp(m′s_uv_tfbf_b_tvbv[i′] * zs[j]) for i′ in eachindex(P_uv_tfbf_b_tvbv)])) ./ f ./ ρ₀
end

##
fig = Figure(resolution=(1000, 500))

axv = Axis(fig[1, 1], title="v")
axg = Axis(fig[1, 2], title="Geostrophic v")

vlim = (minimum([minimum(vs_uv_tfbf_b_tvbv), minimum(v_gs_uv_tfbf_b_tvbv)]), 
        maximum([maximum(vs_uv_tfbf_b_tvbv), maximum(v_gs_uv_tfbf_b_tvbv)]))

heatmap!(axv, xs, zs, vs_uv_tfbf_b_tvbv, colorrange=vlim)
heatmap!(axg, xs, zs, v_gs_uv_tfbf_b_tvbv, colorrange=vlim)

Label(fig[0, :], "Free-slip, constant buoyancy", font=:bold)

# heatmap!(axv, xs, zs, vs_uv_tfbf_b_tvbv)
display(fig)
save("v_v_g_uv_tfbf_b_tvbv.png", fig, px_per_unit=4)
##
# save("Output/Rac_k_uv_tfbf_b_tvbv.png", fig_uv_tfbf_b_tvbv, px_per_unit=4)

nonzero_m′s_uv_tfbf_b_tvbv, nonzero_eigvec_ind_uv_tfbf_b_tvbv = find_zero_eigenvalue(A_eigen_uv_tfbf_b_tvbv, m′s_uv_tfbf_b_tvbv)

##
# uv: top value, bottom value, b: top value, bottom value
bcs_type_uv_tvbv_b_tvbv = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tvbv_b_tvbv = 3:0.01:4
res_Ra_k_uv_tvbv_b_tvbv, fig_uv_tvbv_b_tvbv, A_uv_tvbv_b_tvbv, A_eigen_uv_tvbv_b_tvbv, k′_uv_tvbv_b_tvbv, m′s_uv_tvbv_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tvbv, ms_eq, ks_uv_tvbv_b_tvbv, Ta, Pr; Ra_min=0, Ra_max=10000)
                              
# save("Output/Rac_k_uv_tvbv_b_tvbv.png", fig_uv_tvbv_b_tvbv, px_per_unit=4)

nonzero_m′s_uv_tvbv_b_tvbv, nonzero_eigvec_ind_uv_tvbv_b_tvbv = find_zero_eigenvalue(A_eigen_uv_tvbv_b_tvbv, m′s_uv_tvbv_b_tvbv)

##
# uv: top flux, bottom value, b: top value, bottom value
bcs_type_uv_tfbv_b_tvbv = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tfbv_b_tvbv = 2:0.01:4
res_Ra_k_uv_tfbv_b_tvbv, fig_uv_tfbv_b_tvbv, A_uv_tfbv_b_tvbv, A_eigen_uv_tfbv_b_tvbv, k′_uv_tfbv_b_tvbv, m′s_uv_tfbv_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tvbv, ms_eq, ks_uv_tfbv_b_tvbv, 1000, Pr; Ra_min=Ta*1.2, Ra_max=1e4)
                              
# save("Output/Rac_k_uv_tfbv_b_tvbv.png", fig_uv_tfbv_b_tvbv, px_per_unit=4)

nonzero_m′s_uv_tfbv_b_tvbv, nonzero_eigvec_ind_uv_tfbv_b_tvbv = find_zero_eigenvalue(A_eigen_uv_tfbv_b_tvbv, m′s_uv_tfbv_b_tvbv)

##
# uv: top flux, bottom flux, b: top flux, bottom flux
bcs_type_uv_tfbf_b_tfbf = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))
                              
ks_uv_tfbf_b_tfbf = 0.01:0.01:5
res_Ra_k_uv_tfbf_b_tfbf, fig_uv_tfbf_b_tfbf, A_uv_tfbf_b_tfbf, A_eigen_uv_tfbf_b_tfbf, k′_uv_tfbf_b_tfbf, m′s_uv_tfbf_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tfbf, ms_eq, ks_uv_tfbf_b_tfbf, Ta, Pr; Ra_min=Ta*1.01, Ra_max=1500)

# save("Output/Rac_k_uv_tfbf_b_tfbf.png", fig_uv_tfbf_b_tfbf, px_per_unit=4)

RHS_uv_tfbf_b_tfbf = [0, 0, 0, -res_Ra_k_uv_tfbf_b_tfbf.minimum, 0, 0, 0, -res_Ra_k_uv_tfbf_b_tfbf.minimum]

P_uv_tfbf_b_tfbf = A_uv_tfbf_b_tfbf \ RHS_uv_tfbf_b_tfbf

coeffs_uv_tfbf_b_tfbf = compute_coefficients(k′_uv_tfbf_b_tfbf, m′s_uv_tfbf_b_tfbf; Ra=res_Ra_k_uv_tfbf_b_tfbf.minimum, Ta=Ta, Pr=Pr, ρ₀=ρ₀, H=H, ν=ν)

V_uv_tfbf_b_tfbf = [coeffs_uv_tfbf_b_tfbf[i][2] * P_uv_tfbf_b_tfbf[i] for i in eachindex(P_uv_tfbf_b_tfbf)]

zs = -H/2:0.01:H/2
xs = 0:0.01:4H

vs_uv_tfbf_b_tfbf = zeros(length(xs), length(zs))
for i in eachindex(xs), j in eachindex(zs)
    vs_uv_tfbf_b_tfbf[i, j] = real(sum([V_uv_tfbf_b_tfbf[i′] * exp(m′s_uv_tfbf_b_tfbf[i′] * zs[j]) for i′ in eachindex(V_uv_tfbf_b_tfbf)]) * exp(im * k′_uv_tfbf_b_tfbf * xs[i]))
end

f = √(Ta * ν ^ 2 / H ^ 4)

v_gs_uv_tfbf_b_tfbf = zeros(length(xs), length(zs))
for i in eachindex(xs), j in eachindex(zs)
    v_gs_uv_tfbf_b_tfbf[i, j] = real(im * k′_uv_tfbf_b_tfbf * exp(im * k′_uv_tfbf_b_tfbf * xs[i]) * 
                                   sum([P_uv_tfbf_b_tfbf[i′] * exp(m′s_uv_tfbf_b_tfbf[i′] * zs[j]) for i′ in eachindex(P_uv_tfbf_b_tfbf)])) ./ f ./ ρ₀
end

##
fig = Figure(resolution=(1000, 500))

axv = Axis(fig[1, 1], title="v")
axg = Axis(fig[1, 2], title="Geostrophic v")

vlim = (minimum([minimum(vs_uv_tfbf_b_tfbf), minimum(v_gs_uv_tfbf_b_tfbf)]), 
        maximum([maximum(vs_uv_tfbf_b_tfbf), maximum(v_gs_uv_tfbf_b_tfbf)]))

heatmap!(axv, xs, zs, vs_uv_tfbf_b_tfbf, colorrange=vlim)
heatmap!(axg, xs, zs, v_gs_uv_tfbf_b_tfbf, colorrange=vlim)

Label(fig[0, :], "Free-slip, constant buoyancy flux", font=:bold)

# heatmap!(axv, xs, zs, vs_uv_tfbf_b_tfbf)
display(fig)
save("v_v_g_uv_tfbf_b_tfbf.png", fig, px_per_unit=4)
##

nonzero_m′s_uv_tfbf_b_tfbf, nonzero_eigvec_ind_uv_tfbf_b_tfbf = find_zero_eigenvalue(A_eigen_uv_tfbf_b_tfbf, m′s_uv_tfbf_b_tfbf)

##
# uv: top value, bottom value, b: top flux, bottom flux
bcs_type_uv_tvbv_b_tfbf = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tvbv_b_tfbf = 1e-6:0.01:4.5
res_Ra_k_uv_tvbv_b_tfbf, fig_uv_tvbv_b_tfbf, A_uv_tvbv_b_tfbf, A_eigen_uv_tvbv_b_tfbf, k′_uv_tvbv_b_tfbf, m′s_uv_tvbv_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tfbf, ms_eq, ks_uv_tvbv_b_tfbf, Ta, Pr; Ra_min=1000, Ra_max=1500)
                              
# save("Output/Rac_k_uv_tvbv_b_tfbf.png", fig_uv_tvbv_b_tfbf, px_per_unit=4)

nonzero_m′s_uv_tvbv_b_tfbf, nonzero_eigvec_ind_uv_tvbv_b_tfbf = find_zero_eigenvalue(A_eigen_uv_tvbv_b_tfbf, m′s_uv_tvbv_b_tfbf)

b = [0, 0, 0, 0, 0, 0, 0, res_Ra_k_uv_tvbv_b_tfbf.minimum]

S = res_Ra_k_uv_tvbv_b_tfbf.minimum
# x = A_uv_tvbv_b_tfbf \ b
x = pinv(A_uv_tvbv_b_tfbf) * b

# coeffs = compute_coefficients(res_Ra_k_uv_tvbv_b_tfbf.minimizer, m′s_uv_tvbv_b_tfbf, Ra=S, Ta=1000, Pr=1, ρ₀=1000, H=1, ν=1)
coeffs = compute_coefficients(0, m′s_uv_tvbv_b_tfbf, Ra=S, Ta=1000, Pr=1, ρ₀=1000, H=1, ν=1)
z = -0.25

u_sum = sum([coeffs[i][1] * x[i] * exp(m′s_uv_tvbv_b_tfbf[i] * z) for i in 1:length(x)])
v_sum = sum([coeffs[i][2] * x[i] * exp(m′s_uv_tvbv_b_tfbf[i] * z) for i in 1:length(x)])

##
# uv: top flux, bottom value, b: top flux, bottom flux
bcs_type_uv_tfbv_b_tfbf = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tfbv_b_tfbf = 0.001:0.01:4.5
res_Ra_k_uv_tfbv_b_tfbf, fig_uv_tfbv_b_tfbf, A_uv_tfbv_b_tfbf, A_eigen_uv_tfbv_b_tfbf, k′_uv_tfbv_b_tfbf, m′s_uv_tfbv_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tfbf, ms_eq, ks_uv_tfbv_b_tfbf, Ta, Pr; Ra_min=100, Ra_max=1700)
                              
# save("Output/Rac_k_uv_tfbv_b_tfbf.png", fig_uv_tfbv_b_tfbf, px_per_unit=4)

nonzero_m′s_uv_tfbv_b_tfbf, nonzero_eigvec_ind_uv_tfbv_b_tfbf = find_zero_eigenvalue(A_eigen_uv_tfbv_b_tfbf, m′s_uv_tfbv_b_tfbf)

##
# uv: top flux, bottom flux, b: top value, bottom flux
bcs_type_uv_tfbf_b_tvbf = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=FluxBoundaryCondition))
                              
ks_uv_tfbf_b_tvbf = 2:0.01:4
res_Ra_k_uv_tfbf_b_tvbf, fig_uv_tfbf_b_tvbf, A_uv_tfbf_b_tvbf, A_eigen_uv_tfbf_b_tvbf, k′_uv_tfbf_b_tvbf, m′s_uv_tfbf_b_tvbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tvbf, ms_eq, ks_uv_tfbf_b_tvbf, Ta, Pr; Ra_min=Ta*1.01, Ra_max=2e3)

# save("Output/Rac_k_uv_tfbf_b_tvbf.png", fig_uv_tfbf_b_tvbf, px_per_unit=4)

nonzero_m′s_uv_tfbf_b_tvbf, nonzero_eigvec_ind_uv_tfbf_b_tvbf = find_zero_eigenvalue(A_eigen_uv_tfbf_b_tvbf, m′s_uv_tfbf_b_tvbf)

##
# uv: top value, bottom value, b: top flux, bottom value
bcs_type_uv_tvbv_b_tfbv = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tvbv_b_tfbv = 2:0.01:4
res_Ra_k_uv_tvbv_b_tfbv, fig_uv_tvbv_b_tfbv, A_uv_tvbv_b_tfbv, A_eigen_uv_tvbv_b_tfbv, k′_uv_tvbv_b_tfbv, m′s_uv_tvbv_b_tfbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tfbv, ms_eq, ks_uv_tvbv_b_tfbv, Ta, Pr; Ra_min=Ta, Ra_max=1e4)
                              
# save("Output/Rac_k_uv_tvbv_b_tfbv.png", fig_uv_tvbv_b_tfbv, px_per_unit=4)

nonzero_m′s_uv_tvbv_b_tfbv, nonzero_eigvec_ind_uv_tvbv_b_tfbv = find_zero_eigenvalue(A_eigen_uv_tvbv_b_tfbv, m′s_uv_tvbv_b_tfbv)

##
# uv: top flux, bottom value, b: top flux, bottom value
bcs_type_uv_tfbv_b_tfbv = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tfbv_b_tfbv = 2.5:0.01:3.5
res_Ra_k_uv_tfbv_b_tfbv, fig_uv_tfbv_b_tfbv, A_uv_tfbv_b_tfbv, A_eigen_uv_tfbv_b_tfbv, k′_uv_tfbv_b_tfbv, m′s_uv_tfbv_b_tfbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tfbv, ms_eq, ks_uv_tfbv_b_tfbv, Ta, Pr; Ra_min=10, Ra_max=2e3)
                              
# save("Output/Rac_k_uv_tfbv_b_tfbv.png", fig_uv_tfbv_b_tfbv, px_per_unit=4)

nonzero_m′s_uv_tfbv_b_tfbv, nonzero_eigvec_ind_uv_tfbv_b_tfbv = find_zero_eigenvalue(A_eigen_uv_tfbv_b_tfbv, m′s_uv_tfbv_b_tfbv)

##
# uv: top flux, bottom value, b: top value, bottom flux
bcs_type_uv_tfbv_b_tvbf = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tfbv_b_tvbf = 2:0.01:4
res_Ra_k_uv_tfbv_b_tvbf, fig_uv_tfbv_b_tvbf, A_uv_tfbv_b_tvbf, A_eigen_uv_tfbv_b_tvbf, k′_uv_tfbv_b_tvbf, m′s_uv_tfbv_b_tvbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tvbf, ms_eq, ks_uv_tfbv_b_tvbf, Ta, Pr; Ra_min=Ta*1.2, Ra_max=1e4)
                              
# save("Output/Rac_k_uv_tfbv_b_tvbf.png", fig_uv_tfbv_b_tvbf, px_per_unit=4)

nonzero_m′s_uv_tfbv_b_tvbf, nonzero_eigvec_ind_uv_tfbv_b_tvbf = find_zero_eigenvalue(A_eigen_uv_tfbv_b_tvbf, m′s_uv_tfbv_b_tvbf)

##
