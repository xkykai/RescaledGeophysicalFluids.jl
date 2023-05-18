using SymPy
using LinearAlgebra
using Roots
using CairoMakie
using Optim

m_sym, k_sym, Ta_sym, Ra_sym, α_sym = Sym("m k Ta Ra alpha")
expr = (m_sym^2 - k_sym^2)^3 + Ta_sym * m_sym^2 + Ra_sym * k_sym^2
sol = solve(expr, m_sym)

# k = Complex(2π / 10)
k = Complex(3.71)
Ta = 1000
Ra = 1676

α = 1 / 4
Pr = 1
ν = 1
H = 1

κ = ν / Pr
f = √(Ta * ν ^ 2 / H ^ 4)
S = Ra * ν * κ / H ^ 4
ρ₀ = 1000

m₁⁺_eq, m₁⁻_eq, m₂⁺_eq, m₂⁻_eq, m₃⁺_eq, m₃⁻_eq = [lambdify(sol[i], [k_sym, Ra_sym, Ta_sym]) for i in 1:6]

ms_eq = [m₁⁺_eq, m₁⁻_eq, m₂⁺_eq, m₂⁻_eq, m₃⁺_eq, m₃⁻_eq]

m₀⁺, m₀⁻ = k, -k
m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]

ms = [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]

function compute_coefficients(k, ms, ν, κ, f, S, ρ₀)
    us = [zeros(ComplexF64, 4) for i in 1:length(ms)]
    for (i, m) in pairs(ms)
        A = [ν*(k^2 - m^2) -f 0 0;
            f ν*(k^2 - m^2) 0 0;
            0 0 ν*(k^2 - m^2) -1;
            0 0 S κ*(k^2 - m^2);]

        b = [-im*k/ρ₀, 0, -m/ρ₀, 0]

        us[i] = A \ b
    end
    return us
end

us = compute_coefficients(k, ms, ν, κ, f, S, ρ₀)

struct U_bc{BC, C}
    bc :: BC
    coef :: C
end

struct V_bc{BC, C}
    bc :: BC
    coef :: C
end

struct W_bc{BC, C}
    bc :: BC
    coef :: C
end

struct B_bc{BC, C}
    bc :: BC
    coef :: C
end

struct FluxBoundaryCondition{Z}
    z :: Z
end

struct ValueBoundaryCondition{Z}
    z :: Z
end

function U_bc(bc::FluxBoundaryCondition, us, ms)
    coef = [ms[i] * us[i][1] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return U_bc(bc, coef)
end

function V_bc(bc::FluxBoundaryCondition, us, ms)
    coef = [ms[i] * us[i][2] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return V_bc(bc, coef)
end

function W_bc(bc::FluxBoundaryCondition, us, ms)
    coef = [ms[i] * us[i][3] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return W_bc(bc, coef)
end

function B_bc(bc::FluxBoundaryCondition, us, ms)
    coef = [ms[i] * us[i][4] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return B_bc(bc, coef)
end

function U_bc(bc::ValueBoundaryCondition, us, ms)
    coef = [us[i][1] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return U_bc(bc, coef)
end

function V_bc(bc::ValueBoundaryCondition, us, ms)
    coef = [us[i][2] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return V_bc(bc, coef)
end

function W_bc(bc::ValueBoundaryCondition, us, ms)
    coef = [us[i][3] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return W_bc(bc, coef)
end

function B_bc(bc::ValueBoundaryCondition, us, ms)
    coef = [us[i][4] * exp(ms[i] * bc.z) for i in eachindex(ms) ]
    return B_bc(bc, coef)
end

uv_bc_top = FluxBoundaryCondition(H/2)
uv_bc_bot = FluxBoundaryCondition(-H/2)

w_bc_top = ValueBoundaryCondition(H/2)
w_bc_bot = ValueBoundaryCondition(-H/2)

b_bc_top = ValueBoundaryCondition(H/2)
b_bc_bot = ValueBoundaryCondition(-H/2)

function build_bc_matrix(uv_bc_top, uv_bc_bot, w_bc_top, w_bc_bot, b_bc_top, b_bc_bot, us, ms)
    U_bc_top = U_bc(uv_bc_top, us, ms)
    V_bc_top = V_bc(uv_bc_top, us, ms)
    W_bc_top = W_bc(w_bc_top, us, ms)
    B_bc_top = B_bc(b_bc_top, us, ms)

    U_bc_bot = U_bc(uv_bc_bot, us, ms)
    V_bc_bot = V_bc(uv_bc_bot, us, ms)
    W_bc_bot = W_bc(w_bc_bot, us, ms)
    B_bc_bot = B_bc(b_bc_bot, us, ms)
    return transpose(hcat([U_bc_top.coef, V_bc_top.coef, W_bc_top.coef, B_bc_top.coef, U_bc_bot.coef, V_bc_bot.coef, W_bc_bot.coef, B_bc_bot.coef]...))
end

function det_bc_matrix(k, Ra, Ta, Pr)
    ms_eq = [m₁⁺_eq, m₁⁻_eq, m₂⁺_eq, m₂⁻_eq, m₃⁺_eq, m₃⁻_eq]

    m₀⁺, m₀⁻ = k, -k
    m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]

    ms = [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]

    ν = 1
    H = 1

    κ = ν / Pr
    f = √(Ta * ν ^ 2 / H ^ 4)
    S = Ra * ν * κ / H ^ 4
    ρ₀ = 1000

    us = compute_coefficients(k, ms, ν, κ, f, S, ρ₀)

    uv_bc_top = ValueBoundaryCondition(H/2)
    uv_bc_bot = ValueBoundaryCondition(-H/2)

    w_bc_top = ValueBoundaryCondition(H/2)
    w_bc_bot = ValueBoundaryCondition(-H/2)

    b_bc_top = ValueBoundaryCondition(H/2)
    b_bc_bot = ValueBoundaryCondition(-H/2)

    A = build_bc_matrix(uv_bc_top, uv_bc_bot, w_bc_top, w_bc_bot, b_bc_top, b_bc_bot, us, ms)

    return logabsdet(A)[1]
end

det_bc_matrix(Ra) = det_bc_matrix(k, Ra, Ta, Pr)

function find_critical_Ra(k, bcs_type, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)
    k = Complex(k)
    ms_eq = [m₁⁺_eq, m₁⁻_eq, m₂⁺_eq, m₂⁻_eq, m₃⁺_eq, m₃⁻_eq]

    m₀⁺, m₀⁻ = k, -k

    κ = ν / Pr
    f = √(Ta * ν ^ 2 / H ^ 4)

    A = build_bc_matrix(uv_bc_top, uv_bc_bot, w_bc_top, w_bc_bot, b_bc_top, b_bc_bot, us, ms)

    function find_logdet_Ra(Ra)
        S = Ra * ν * κ / H ^ 4
        m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]
        ms = [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]

        bcs = (; uv = (; top=bcs_type.uv.top(H/2), bot=bcs_type.uv.bot(-H/2)),
                  w = (; top=bcs_type.w.top(H/2), bot=bcs_type.w.bot(-H/2)),
                  b = (; top=bcs_type.b.top(H/2), bot=bcs_type.b.bot(-H/2)))

        coeffs = compute_coefficients(k, ms, ν, κ, f, S, ρ₀)
        A = build_bc_matrix(bcs.uv.top, bcs.uv.bot, bcs.w.top, bcs.w.bot, bcs.b.top, bcs.b.bot, coeffs, ms)
        return logabsdet(A)[1]
    end

    res_Ra = optimize(find_logdet_Ra, Ra_min, Ra_max)
    return res_Ra.minimizer
end

function find_critical_Ra_k(k_min, k_max, bcs_type, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)
    objective(k) = find_critical_Ra(k, bcs_type, Ta, Pr, Ra_min, Ra_max; ν=ν, H=H, ρ₀=ρ₀)

    res_k = optimize(objective, k_min, k_max)
    return res_k
end

function build_bc_matrix(k, bcs_type, Ta, Pr, Ra; ν=1, H=1, ρ₀=1000)
    k = Complex(k)
    ms_eq = [m₁⁺_eq, m₁⁻_eq, m₂⁺_eq, m₂⁻_eq, m₃⁺_eq, m₃⁻_eq]

    m₀⁺, m₀⁻ = k, -k
    m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]

    κ = ν / Pr
    f = √(Ta * ν ^ 2 / H ^ 4)
    S = Ra * ν * κ / H ^ 4
    
    ms = [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]

    bcs = (; uv = (; top=bcs_type.uv.top(H/2), bot=bcs_type.uv.bot(-H/2)),
              w = (; top=bcs_type.w.top(H/2), bot=bcs_type.w.bot(-H/2)),
              b = (; top=bcs_type.b.top(H/2), bot=bcs_type.b.bot(-H/2)))

    coeffs = compute_coefficients(k, ms, ν, κ, f, S, ρ₀)

    A = build_bc_matrix(bcs.uv.top, bcs.uv.bot, bcs.w.top, bcs.w.bot, bcs.b.top, bcs.b.bot, coeffs, ms)
    return A
end

##
function find_plot_critical_Ra_k_A(bcs_type, ks, Ta, Pr; Ra_min, Ra_max, ν=1, H=1, ρ₀=1000)
    @info "Finding critical Ra for each k..."
    res_Ras = [find_critical_Ra(k, bcs_type, Ta, Pr, Ra_min, Ra_max, ν=ν, H=H, ρ₀=ρ₀) for k in ks]

    @info "Finding critical and k..."
    res_Ra_k = find_critical_Ra_k(ks[1], ks[end], bcs_type, Ta, Pr, Ra_min, Ra_max)

    k′ = res_Ra_k.minimizer
    Ra′ = res_Ra_k.minimum

    @info "Plotting results..."
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="Critical Ra", title="uv: $(bcs_type.uv), \n w: $(bcs_type.w), \n b: $(bcs_type.b)")
    lines!(ax, ks, res_Ras)
    scatter!(ax, [k′], [Ra′], color=:black, label="k = $(round(k′, digits=2)), Ra = $(round(Ra′, digits=2))")
    axislegend(ax)
    display(fig)

    @info "Building matrix"
    A = build_bc_matrix(k′, bcs_type, Ta, Pr, Ra′; ν=1, H=1, ρ₀=1000)
    A_eigen = eigen(A)

    m₀⁺′, m₀⁻′ = k′, -k′
    m₁⁺′, m₁⁻′, m₂⁺′, m₂⁻′, m₃⁺′, m₃⁻′ = [ms_eq[i](Complex(k′), Ra′, Ta) for i in 1:6]

    m′s = [m₀⁺′, m₀⁻′, m₁⁺′, m₁⁻′, m₂⁺′, m₂⁻′, m₃⁺′, m₃⁻′]

    return res_Ra_k, fig, A, A_eigen, k′, m′s
end
##
# uv: top flux, bottom flux, b: top value, bottom value
bcs_type_uv_tfbf_b_tvbv = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))
                              
ks_uv_tfbf_b_tvbv = 1.5:0.01:5
res_Ra_k_uv_tfbf_b_tvbv, fig_uv_tfbf_b_tvbv, A_uv_tfbf_b_tvbv, A_eigen_uv_tfbf_b_tvbv, k′_uv_tfbf_b_tvbv, m′s_uv_tfbf_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tvbv, ks_uv_tfbf_b_tvbv, Ta, Pr; Ra_min=Ta*1.5, Ra_max=1e4)

save("Output/Rac_k_uv_tfbf_b_tvbv.png", fig_uv_tfbf_b_tvbv, px_per_unit=4)

##
# uv: top value, bottom value, b: top value, bottom value
bcs_type_uv_tvbv_b_tvbv = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tvbv_b_tvbv = 1.5:0.01:5
res_Ra_k_uv_tvbv_b_tvbv, fig_uv_tvbv_b_tvbv, A_uv_tvbv_b_tvbv, A_eigen_uv_tvbv_b_tvbv, k′_uv_tvbv_b_tvbv, m′s_uv_tvbv_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tvbv, ks_uv_tvbv_b_tvbv, Ta, Pr; Ra_min=Ta*1.5, Ra_max=1e4)
                              
save("Output/Rac_k_uv_tvbv_b_tvbv.png", fig_uv_tvbv_b_tvbv, px_per_unit=4)
##
# uv: top flux, bottom value, b: top value, bottom value
bcs_type_uv_tfbv_b_tvbv = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tfbv_b_tvbv = 1.5:0.01:4.5
res_Ra_k_uv_tfbv_b_tvbv, fig_uv_tfbv_b_tvbv, A_uv_tfbv_b_tvbv, A_eigen_uv_tfbv_b_tvbv, k′_uv_tfbv_b_tvbv, m′s_uv_tfbv_b_tvbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tvbv, ks_uv_tfbv_b_tvbv, Ta, Pr; Ra_min=Ta*1.5, Ra_max=1e4)
                              
save("Output/Rac_k_uv_tfbv_b_tvbv.png", fig_uv_tfbv_b_tvbv, px_per_unit=4)
##
# uv: top flux, bottom flux, b: top flux, bottom flux
bcs_type_uv_tfbf_b_tfbf = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))
                              
ks_uv_tfbf_b_tfbf = 3:0.01:5
res_Ra_k_uv_tfbf_b_tfbf, fig_uv_tfbf_b_tfbf, A_uv_tfbf_b_tfbf, A_eigen_uv_tfbf_b_tfbf, k′_uv_tfbf_b_tfbf, m′s_uv_tfbf_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tfbf, ks_uv_tfbf_b_tfbf, Ta, Pr; Ra_min=Ta*1.5, Ra_max=1e4)

save("Output/Rac_k_uv_tfbf_b_tfbf.png", fig_uv_tfbf_b_tfbf, px_per_unit=4)
##
# uv: top value, bottom value, b: top flux, bottom flux
bcs_type_uv_tvbv_b_tfbf = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tvbv_b_tfbf = 3:0.01:4.5
res_Ra_k_uv_tvbv_b_tfbf, fig_uv_tvbv_b_tfbf, A_uv_tvbv_b_tfbf, A_eigen_uv_tvbv_b_tfbf, k′_uv_tvbv_b_tfbf, m′s_uv_tvbv_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tfbf, ks_uv_tvbv_b_tfbf, Ta, Pr; Ra_min=Ta*1.2, Ra_max=2e4)
                              
save("Output/Rac_k_uv_tvbv_b_tfbf.png", fig_uv_tvbv_b_tfbf, px_per_unit=4)
##
# uv: top flux, bottom value, b: top flux, bottom flux
bcs_type_uv_tfbv_b_tfbf = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tfbv_b_tfbf = 3:0.01:4.5
res_Ra_k_uv_tfbv_b_tfbf, fig_uv_tfbv_b_tfbf, A_uv_tfbv_b_tfbf, A_eigen_uv_tfbv_b_tfbf, k′_uv_tfbv_b_tfbf, m′s_uv_tfbv_b_tfbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tfbf, ks_uv_tfbv_b_tfbf, Ta, Pr; Ra_min=Ta*1.5, Ra_max=2e4)
                              
save("Output/Rac_k_uv_tfbv_b_tfbf.png", fig_uv_tvbv_b_tfbf, px_per_unit=4)
##
# uv: top flux, bottom flux, b: top value, bottom flux
bcs_type_uv_tfbf_b_tvbf = (; uv = (; top=FluxBoundaryCondition, bot=FluxBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=FluxBoundaryCondition))
                              
ks_uv_tfbf_b_tvbf = 3.5:0.01:5
res_Ra_k_uv_tfbf_b_tvbf, fig_uv_tfbf_b_tvbf, A_uv_tfbf_b_tvbf, A_eigen_uv_tfbf_b_tvbf, k′_uv_tfbf_b_tvbf, m′s_uv_tfbf_b_tvbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbf_b_tvbf, ks_uv_tfbf_b_tvbf, Ta, Pr; Ra_min=Ta*1.5, Ra_max=2e4)

save("Output/Rac_k_uv_tfbf_b_tvbf.png", fig_uv_tfbf_b_tvbf, px_per_unit=4)
##
# uv: top value, bottom value, b: top flux, bottom value
bcs_type_uv_tvbv_b_tfbv = (; uv = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tvbv_b_tfbv = 3.5:0.01:5
res_Ra_k_uv_tvbv_b_tfbv, fig_uv_tvbv_b_tfbv, A_uv_tvbv_b_tfbv, A_eigen_uv_tvbv_b_tfbv, k′_uv_tvbv_b_tfbv, m′s_uv_tvbv_b_tfbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tvbv_b_tfbv, ks_uv_tvbv_b_tfbv, Ta, Pr; Ra_min=Ta, Ra_max=5e4)
                              
save("Output/Rac_k_uv_tvbv_b_tfbv.png", fig_uv_tvbv_b_tfbv, px_per_unit=4)
##
# uv: top flux, bottom value, b: top flux, bottom value
bcs_type_uv_tfbv_b_tfbv = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition))

ks_uv_tfbv_b_tfbv = 3:0.01:5.5
res_Ra_k_uv_tfbv_b_tfbv, fig_uv_tfbv_b_tfbv, A_uv_tfbv_b_tfbv, A_eigen_uv_tfbv_b_tfbv, k′_uv_tfbv_b_tfbv, m′s_uv_tfbv_b_tfbv = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tfbv, ks_uv_tfbv_b_tfbv, Ta, Pr; Ra_min=Ta, Ra_max=5e4)
                              
save("Output/Rac_k_uv_tfbv_b_tfbv.png", fig_uv_tfbv_b_tfbv, px_per_unit=4)
##
# uv: top flux, bottom value, b: top value, bottom flux
bcs_type_uv_tfbv_b_tvbf = (; uv = (; top=FluxBoundaryCondition, bot=ValueBoundaryCondition), 
                              w = (; top=ValueBoundaryCondition, bot=ValueBoundaryCondition),
                              b = (; top=ValueBoundaryCondition, bot=FluxBoundaryCondition))

ks_uv_tfbv_b_tvbf = 1:0.01:3.5
res_Ra_k_uv_tfbv_b_tvbf, fig_uv_tfbv_b_tvbf, A_uv_tfbv_b_tvbf, A_eigen_uv_tfbv_b_tvbf, k′_uv_tfbv_b_tvbf, m′s_uv_tfbv_b_tvbf = find_plot_critical_Ra_k_A(
    bcs_type_uv_tfbv_b_tvbf, ks_uv_tfbv_b_tvbf, Ta, Pr; Ra_min=Ta*1.2, Ra_max=1e4)
                              
save("Output/Rac_k_uv_tfbv_b_tvbf.png", fig_uv_tfbv_b_tvbf, px_per_unit=4)
##
