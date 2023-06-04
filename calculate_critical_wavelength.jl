using Oceananigans
using CairoMakie
using RescaledGeophysicalFluids

# FILE_DIR = "Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_1800_Ta_1000_alpha_0.125_Nz_64"
FILE_DIR = "Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_30000.0_Ta_100000.0_alpha_0.125_Nz_64"

b_data = FieldTimeSeries(joinpath(FILE_DIR, "instantaneous_fields.jld2"), "b")

xb, yb, zb = nodes(b_data)
bs = interior(b_data[end], :, 1, Int(floor(length(zb) / 2)))

kc, kc_neighbourhood = calculate_critical_k(bs, xb)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, xb, bs)
display(fig)

# fig = Figure()
# ax = Axis(fig[1, 1])
# lines!(ax, ks, abs.(F))
# display(fig)

@info "Critical k = $(kc) in the neighbourhood of $(kc_neighbourhood)"