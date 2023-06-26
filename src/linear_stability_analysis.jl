function compute_coefficients(k, ms; Ra, Ta, Pr, ρ₀, H, ν)
    κ = ν / Pr
    f = √(Ta * ν ^ 2 / H ^ 4)
    S = Ra * ν * κ / H ^ 4

    us = [zeros(ComplexF64, 4) for i in 1:length(ms)]
    for (i, m) in pairs(ms)
        A = [ν*(k^2 - m^2) -f 0 0;
            f ν*(k^2 - m^2) 0 0;
            0 0 ν*(k^2 - m^2) -1;
            0 0 -S κ*(k^2 - m^2);]

        b = [-im*k/ρ₀, 0, -m/ρ₀, 0]

        us[i] .= A \ b
    end
    return us
end

function compute_coefficients_ω(k, ω, ms; Ra, Ta, Pr, ρ₀, H, ν)
    κ = ν / Pr
    f = √(Ta * ν ^ 2 / H ^ 4)
    S = Ra * ν * κ / H ^ 4
    ∂t = im * ω

    us = [zeros(ComplexF64, 4) for i in 1:length(ms)]
    for (i, m) in pairs(ms)
        A = [∂t+ν*(k^2 - m^2) -f 0 0;
            f ∂t+ν*(k^2 - m^2) 0 0;
            0 0 ∂t+ν*(k^2 - m^2) -1;
            0 0 -S ∂t+κ*(k^2 - m^2);]

        b = [-im*k/ρ₀, 0, -m/ρ₀, 0]
        us[i] .= A \ b
    end
    return us
end
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

function build_bc_matrix(uv_bc_top, uv_bc_bot, w_bc_top, w_bc_bot, b_bc_top, b_bc_bot, us, ms)
    U_bc_top = U_bc(uv_bc_top, us, ms)
    W_bc_top = W_bc(w_bc_top, us, ms)
    B_bc_top = B_bc(b_bc_top, us, ms)
    U_bc_bot = U_bc(uv_bc_bot, us, ms)
    W_bc_bot = W_bc(w_bc_bot, us, ms)
    B_bc_bot = B_bc(b_bc_bot, us, ms)
    
    if length(ms) == 8
        V_bc_top = V_bc(uv_bc_top, us, ms)
        V_bc_bot = V_bc(uv_bc_bot, us, ms)
        return transpose(hcat([U_bc_top.coef, V_bc_top.coef, W_bc_top.coef, B_bc_top.coef, U_bc_bot.coef, V_bc_bot.coef, W_bc_bot.coef, B_bc_bot.coef]...))
    else
        return transpose(hcat([U_bc_top.coef, W_bc_top.coef, B_bc_top.coef, U_bc_bot.coef, W_bc_bot.coef, B_bc_bot.coef]...))
    end
end

function evaluate_m_expr()
    m_sym, k_sym, Ta_sym, Ra_sym = Sym("m k Ta Ra")
    expr = (m_sym^2 - k_sym^2)^3 + Ta_sym * m_sym^2 + Ra_sym * k_sym^2
    expr = simplify(expr)

    sol = solve(expr, m_sym)
    ms_eq = [lambdify(sol[i], [k_sym, Ra_sym, Ta_sym]) for i in eachindex(sol)]
    return ms_eq
end

function evaluate_m_expr_ω(Ta, Pr)
    m, k, Ra, ∂t = Sym("m k Ra dt")
    ∇² = m^2 - k^2
    expr = (Pr*∂t - ∇²) * (∂t - ∇²)^2 * ∇² + Ta * (Pr*∂t - ∇²) * m^2 + Ra * (∂t - ∇²) * k^2
    expr = simplify(expr)

    sol = solve(expr, m)
    ms_eq = [lambdify(sol[i], [k, Ra, ∂t]) for i in eachindex(sol)]
    return ms_eq
end

function evaluate_m(ms_eq, k; Ra, Ta)
    k = Complex(k)
    m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]

    if Ta != 0
        m₀⁺, m₀⁻ = k, -k
        return [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]
    else
        return [m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]
    end
end

function build_bc_matrix(k, bcs_type, ms_eq, Ta, Pr, Ra; ν=1, H=1, ρ₀=1000)
    k = Complex(k)
    m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻ = [ms_eq[i](k, Ra, Ta) for i in 1:6]

    if Ta != 0
        m₀⁺, m₀⁻ = k, -k
        ms = [m₀⁺, m₀⁻, m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]
    else
        ms = [m₁⁺, m₁⁻, m₂⁺, m₂⁻, m₃⁺, m₃⁻]
    end
    
    bcs = (; uv = (; top=bcs_type.uv.top(H/2), bot=bcs_type.uv.bot(-H/2)),
              w = (; top=bcs_type.w.top(H/2), bot=bcs_type.w.bot(-H/2)),
              b = (; top=bcs_type.b.top(H/2), bot=bcs_type.b.bot(-H/2)))

    coeffs = compute_coefficients(k, ms, Ra=Ra, Ta=Ta, Pr=Pr, ρ₀=ρ₀, H=H, ν=ν)

    A = build_bc_matrix(bcs.uv.top, bcs.uv.bot, bcs.w.top, bcs.w.bot, bcs.b.top, bcs.b.bot, coeffs, ms)
    return A
end

function build_bc_matrix_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra; ν=1, H=1, ρ₀=1000)
    k = Complex(k)
    ∂t = im * ω
    ms = [ms_eq[i](k, Ra, ∂t) for i in eachindex(ms_eq)]

    bcs = (; uv = (; top=bcs_type.uv.top(H/2), bot=bcs_type.uv.bot(-H/2)),
              w = (; top=bcs_type.w.top(H/2), bot=bcs_type.w.bot(-H/2)),
              b = (; top=bcs_type.b.top(H/2), bot=bcs_type.b.bot(-H/2)))

    coeffs = compute_coefficients_ω(k, ω, ms, Ra=Ra, Ta=Ta, Pr=Pr, ρ₀=ρ₀, H=H, ν=ν)

    A = build_bc_matrix(bcs.uv.top, bcs.uv.bot, bcs.w.top, bcs.w.bot, bcs.b.top, bcs.b.bot, coeffs, ms)
    return A
end

function find_logdet_A(k, bcs_type, ms_eq, Ta, Pr, Ra; ν=1, H=1, ρ₀=1000)
    A = build_bc_matrix(k, bcs_type, ms_eq, Ta, Pr, Ra; ν=ν, H=H, ρ₀=ρ₀)
    return logabsdet(A)[1]
end

function find_logdet_A_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra; ν=1, H=1, ρ₀=1000)
    A = build_bc_matrix_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra; ν=ν, H=H, ρ₀=ρ₀)
    return logabsdet(A)[1]
end

function find_critical_Ra(k, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)

    find_logdet_Ra(Ra) = find_logdet_A(k, bcs_type, ms_eq, Ta, Pr, Ra; ν=ν, H=H, ρ₀=ρ₀)

    res_Ra = optimize(find_logdet_Ra, Ra_min, Ra_max)
    return res_Ra.minimizer
end

function find_critical_Ra_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)

    find_logdet_Ra(Ra) = find_logdet_A_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra; ν=ν, H=H, ρ₀=ρ₀)

    res_Ra = optimize(find_logdet_Ra, Ra_min, Ra_max)
    return res_Ra.minimizer
end

function find_critical_Ra_k(k_min, k_max, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)
    objective(k) = find_critical_Ra(k, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=ν, H=H, ρ₀=ρ₀)

    res_k = optimize(objective, k_min, k_max)
    return res_k
end

function find_critical_Ra_k_ω(k_min, k_max, ω, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=1, H=1, ρ₀=1000)
    objective(k) = find_critical_Ra_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max; ν=ν, H=H, ρ₀=ρ₀)

    res_k = optimize(objective, k_min, k_max)
    return res_k
end

function find_plot_critical_Ra_k_A(bcs_type, ms_eq, ks, Ta, Pr; Ra_min, Ra_max, ν=1, H=1, ρ₀=1000)
    @info "Finding critical Ra for each k..."
    res_Ras = [find_critical_Ra(k, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max, ν=ν, H=H, ρ₀=ρ₀) for k in ks]

    @info "Finding critical and k..."
    res_Ra_k = find_critical_Ra_k(ks[1], ks[end], bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max)

    k′ = res_Ra_k.minimizer
    Ra′ = res_Ra_k.minimum

    @info "Plotting results..."
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="Critical Ra", title="uv: $(bcs_type.uv), \n w: $(bcs_type.w), \n b: $(bcs_type.b)")
    lines!(ax, ks, res_Ras)
    scatter!(ax, [k′], [Ra′], color=:black, label="k = $(round(k′, digits=8)), Ra = $(round(Ra′, digits=1)), Ta = $Ta")
    axislegend(ax)
    display(fig)

    @info "Building matrix"
    A = build_bc_matrix(k′, bcs_type, ms_eq, Ta, Pr, Ra′; ν=ν, H=H, ρ₀=ρ₀)
    A_eigen = eigen(A)

    m₁⁺′, m₁⁻′, m₂⁺′, m₂⁻′, m₃⁺′, m₃⁻′ = [ms_eq[i](Complex(k′), Ra′, Ta) for i in 1:6]

    if Ta != 0
        m₀⁺′, m₀⁻′ = k′, -k′
        m′s = [m₀⁺′, m₀⁻′, m₁⁺′, m₁⁻′, m₂⁺′, m₂⁻′, m₃⁺′, m₃⁻′]
    else
        m′s = [m₁⁺′, m₁⁻′, m₂⁺′, m₂⁻′, m₃⁺′, m₃⁻′]
    end

    return res_Ra_k, fig, A, A_eigen, k′, m′s
end

function find_plot_critical_Ra_k_A_ω(bcs_type, ms_eq, ks, ω, Ta, Pr; Ra_min, Ra_max, ν=1, H=1, ρ₀=1000)
    @info "Finding critical Ra for each k..."
    res_Ras = [find_critical_Ra_ω(k, ω, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max, ν=ν, H=H, ρ₀=ρ₀) for k in ks]

    @info "Finding critical and k..."
    res_Ra_k = find_critical_Ra_k_ω(ks[1], ks[end], ω, bcs_type, ms_eq, Ta, Pr, Ra_min, Ra_max)

    k′ = res_Ra_k.minimizer
    Ra′ = res_Ra_k.minimum

    @info "Plotting results..."
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="k", ylabel="Critical Ra", title="uv: $(bcs_type.uv), \n w: $(bcs_type.w), \n b: $(bcs_type.b)")
    lines!(ax, ks, res_Ras)
    scatter!(ax, [k′], [Ra′], color=:black, label="k = $(round(k′, digits=8)), Ra = $(round(Ra′, digits=1)), ω = $ω, Ta = $Ta")
    axislegend(ax)
    display(fig)

    @info "Building matrix"
    A = build_bc_matrix_ω(k′, ω, bcs_type, ms_eq, Ta, Pr, Ra′; ν=ν, H=H, ρ₀=ρ₀)
    A_eigen = eigen(A)

    m′s = [ms_eq[i](Complex(k′), Ra′, im*ω) for i in eachindex(ms_eq)]

    return res_Ra_k, fig, A, A_eigen, k′, m′s
end

function find_zero_eigenvalue(A_eigen::Eigen, ms)
    ϵ = √(eps(typeof(real(A_eigen.values[1]))))
    zero_eigval_ind = isapprox.(A_eigen.values, 0, atol=ϵ)
    zero_eigvec = A_eigen.vectors[:, zero_eigval_ind]
    nonzero_eigvec_ind = .!isapprox.(zero_eigvec, 0, atol=ϵ)
    nonzero_ms = [ms[nonzero_eigvec_ind[:, i]] for i in axes(nonzero_eigvec_ind, 2)]
    return nonzero_ms, nonzero_eigvec_ind
end