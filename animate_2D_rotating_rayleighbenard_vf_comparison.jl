using CairoMakie
using Oceananigans
using Oceananigans.Grids: halo_size
using JLD2
using Statistics

# FILE_DIR_f = "./Data/2D_no_wind_uv_tfbf_b_tfbf_Ra_1136.5_Ta_1000.0_alpha_0.0625_Nz_64"
# FILE_DIR_v = "./Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_1759.0_Ta_1000.0_alpha_0.0625_Nz_64"

# FILE_DIR_f = "./Data/2D_no_wind_uv_tfbf_b_tfbf_Ra_90599.0_Ta_1.0e6_alpha_0.125_Nz_64"
# FILE_DIR_v = "./Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_92260.8_Ta_1.0e6_alpha_0.125_Nz_64"

# FILE_DIR_f = "./Data/2D_no_wind_uv_tfbf_b_tfbf_Ra_1.1e7_Ta_1.0e9_alpha_1.0_Nz_128"
# FILE_DIR_v = "./Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_9.331674098620728e6_Ta_1.0e9_alpha_1.0_Nz_128"

FILE_DIR_f = "./Data/2D_no_wind_uv_tfbf_b_tfbf_Ra_94385.64416955659_Ta_1.0e6_alpha_0.25188489844833484_Nz_64"
FILE_DIR_v = "./Data/2D_no_wind_uv_tfbf_b_tvbv_Ra_96014.66102527594_Ta_1.0e6_alpha_0.25352089021330976_Nz_64"

function load_dataset(FILE_DIR)
    metadata = jldopen("$(FILE_DIR)/instantaneous_timeseries.jld2", "r") do file
        metadata = Dict()
        for key in keys(file["metadata/parameters"])
            metadata[key] = file["metadata/parameters/$(key)"]
        end
        return metadata
    end

    b = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "b")
    w = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
    PV = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "PV")

    B = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "B")
    WB = FieldTimeSeries("$(FILE_DIR)/instantaneous_timeseries.jld2", "WB")
    # ∂x_p = FieldTimeSeries("$(FILE_DIR)/averaged_timeseries.jld2", "∂x_p")
    # ∇²u = FieldTimeSeries("$(FILE_DIR)/averaged_timeseries.jld2", "∇²u")
    # v = FieldTimeSeries("$(FILE_DIR)/averaged_timeseries.jld2", "v")

    ∂x_p = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "∂x_p")
    ∇²u = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "∇²u")
    v = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "v")

    Pr = metadata["prandtl_number"]
    Ra = metadata["rayleigh_number"]
    Lz = b.grid.Lz

    ν = 1
    κ = ν / Pr
    S = Ra * ν * κ / Lz ^ 4
    Nu = WB ./ (κ * S)
    return b, w, PV, B, WB, ∂x_p, ∇²u, v, Nu, metadata
end

function calculate_rms_balance(∂x_p, ∇²u, v, f, ν)
    geos_balance = (f .* interior(v) .- interior(∂x_p)) ./ (f .* interior(v))
    p_balance = interior(∂x_p) ./ (f .* interior(v))
    diff_balance = ν .* interior(∇²u) ./ (f .* interior(v))

    geos_balance_rms = sqrt.(mean(geos_balance .^ 2, dims=(1, 3)))
    p_balance_rms = sqrt.(mean(p_balance .^ 2, dims=(1, 3)))
    diff_balance_rms = sqrt.(mean(diff_balance .^ 2, dims=(1, 3)))

    # geos_balance_rms = replace(geos_balance_rms, NaN => 1)
    # p_balance_rms = replace(p_balance_rms, NaN => 1)
    # diff_balance_rms = replace(diff_balance_rms, NaN => 1)

    return geos_balance_rms, p_balance_rms, diff_balance_rms
end

function calculate_balance(∂x_p, ∇²u, v, f, ν)
    geos_balance = f .* interior(v) .- interior(∂x_p)
    diff_balance = ν .* interior(∇²u)

    geos_balance_ratio_rms = sqrt.(mean((f .* interior(v) .- interior(∂x_p)).^2, dims=(1, 3))) ./ sqrt.(mean(interior(∂x_p).^2, dims=(1, 3)))
    # geos_balance_ratio_rms = replace(geos_balance_ratio_rms, NaN => 1)

    stat_balance = geos_balance .+ diff_balance

    return geos_balance, diff_balance, stat_balance, geos_balance_ratio_rms
end

b_data_f, w_data_f, PV_data_f, B_data_f, WB_data_f, ∂x_p_data_f, ∇²u_data_f, v_data_f, Nu_data_f, metadata_f = load_dataset(FILE_DIR_f)
b_data_v, w_data_v, PV_data_v, B_data_v, WB_data_v, ∂x_p_data_v, ∇²u_data_v, v_data_v, Nu_data_v, metadata_v = load_dataset(FILE_DIR_v)

w_rms_f = sqrt.(mean(interior(w_data_f) .^ 2, dims=(1, 3)))
w_rms_v = sqrt.(mean(interior(w_data_v) .^ 2, dims=(1, 3)))

# w_rms_f = replace(w_rms_f, 0 => 1e-15)
# w_rms_v = replace(w_rms_v, 0 => 1e-15)

# geos_balance_rms_f, p_balance_rms_f, diff_balance_rms_f = calculate_rms_balance(∂x_p_data_f, ∇²u_data_f, v_data_f, metadata_f["coriolis_parameter"], 1)
# geos_balance_rms_v, p_balance_rms_v, diff_balance_rms_v = calculate_rms_balance(∂x_p_data_v, ∇²u_data_v, v_data_v, metadata_v["coriolis_parameter"], 1)

geos_balance_f, diff_balance_f, stat_balance_f, geos_balance_ratio_rms_f = calculate_balance(∂x_p_data_f, ∇²u_data_f, v_data_f, metadata_f["coriolis_parameter"], 1)
geos_balance_v, diff_balance_v, stat_balance_v, geos_balance_ratio_rms_v = calculate_balance(∂x_p_data_v, ∇²u_data_v, v_data_v, metadata_v["coriolis_parameter"], 1)

Lz_f = b_data_f.grid.Lz
Lz_v = b_data_v.grid.Lz

Nt = minimum([length(b_data_f.times), length(b_data_v.times)])

Nx_f = b_data_f.grid.Nx
Nz_f = b_data_f.grid.Nz
xC_f = B_data_f.grid.xᶜᵃᵃ[1:Nx_f]
zC_f = B_data_f.grid.zᵃᵃᶜ[1:Nz_f]
zF_f = WB_data_f.grid.zᵃᵃᶠ[1:Nz_f+1]

Nx_v = b_data_v.grid.Nx
Nz_v = b_data_v.grid.Nz
xC_v = B_data_v.grid.xᶜᵃᵃ[1:Nx_v]
zC_v = B_data_v.grid.zᵃᵃᶜ[1:Nz_v]
zF_v = WB_data_v.grid.zᵃᵃᶠ[1:Nz_v+1]

##
fig = Figure(resolution=(2000, 1100))

n = Observable(1)

Ra_f = metadata_f["rayleigh_number"]
Ra_v = metadata_v["rayleigh_number"]
Ta = metadata_f["taylor_number"]
Pr = metadata_f["prandtl_number"]

axb_f = Axis(fig[1, 1:2], title="b, flux boundary condition, Ra = $(Ra_f)", xlabel="x", ylabel="z")
axb_v = Axis(fig[2, 1:2], title="b, value boundary condition, Ra = $(Ra_v)", xlabel="x", ylabel="z")

axgeos_f = Axis(fig[1, 3:4], title="fv - ∂x(p), flux boundary condition", xlabel="x", ylabel="z")
axgeos_v = Axis(fig[2, 3:4], title="fv - ∂x(p), value boundary condition", xlabel="x", ylabel="z")

axdiff_f = Axis(fig[1, 6:7], title="ν∇²u, flux boundary condition", xlabel="x", ylabel="z")
axdiff_v = Axis(fig[2, 6:7], title="ν∇²u, value boundary condition", xlabel="x", ylabel="z")

axstat_f = Axis(fig[1, 9:10], title="fv - ∂x(p) + ν∇²u, flux boundary condition", xlabel="x", ylabel="z")
axstat_v = Axis(fig[2, 9:10], title="fv - ∂x(p) + ν∇²u, value boundary condition", xlabel="x", ylabel="z")

axPV_f = Axis(fig[3, 3:4], title="PV, ∂x(v) * ∂z(b) - ∂z(v) * ∂x(b) + f * ∂z(b), flux", xlabel="x", ylabel="z")
axPV_v = Axis(fig[4, 3:4], title="PV, ∂x(v) * ∂z(b) - ∂z(v) * ∂x(b) + f * ∂z(b), value", xlabel="x", ylabel="z")

axB = Axis(fig[3, 1:2], title="<b>", xlabel="<b>", ylabel="z")
axNu = Axis(fig[4, 1:2], title="Nu", xlabel="Nu", ylabel="z")

axr = Axis(fig[3,6:11], title="rms[fv - ∂x(p)] / rms[∂x(p)]", xlabel="t", yscale=log10)
axw = Axis(fig[4,6:11], title="rms[w]", xlabel="t", yscale=log)

bn_f = @lift interior(b_data_f[$n], :, 1, :)
bn_v = @lift interior(b_data_v[$n], :, 1, :)

geosn_f = @lift geos_balance_f[:, 1, :, $n]
geosn_v = @lift geos_balance_v[:, 1, :, $n]

diffn_f = @lift diff_balance_f[:, 1, :, $n]
diffn_v = @lift diff_balance_v[:, 1, :, $n]

statn_f = @lift stat_balance_f[:, 1, :, $n]
statn_v = @lift stat_balance_v[:, 1, :, $n]

PVn_f = @lift interior(PV_data_f[$n], :, 1, :)
PVn_v = @lift interior(PV_data_v[$n], :, 1, :)

Bn_f = @lift interior(B_data_f[$n], 1, 1, :) .- interior(B_data_f[$n], 1, 1, 1)
Bn_v = @lift interior(B_data_v[$n], 1, 1, :)

Nun_f = @lift Nu_data_f[1, 1, :, $n]
Nun_v = @lift Nu_data_v[1, 1, :, $n]

time_str = @lift "Free-slip, Ta = $(Ta), Pr = $(Pr), Time = $(round(b_data_f.times[$n], digits=2))"

times = b_data_f.times                  
t = @lift times[$n]

title = Label(fig[0, :], time_str, font=:bold)

blim_f = (minimum(b_data_f), maximum(b_data_f))
blim_v = (minimum(b_data_v), maximum(b_data_v))

Blim = (minimum([minimum(B_data_f), minimum(B_data_v)]), maximum([maximum(B_data_f), maximum(B_data_v)]))
Nulim = (minimum([minimum(Nu_data_f), minimum(Nu_data_v)]), maximum([maximum(Nu_data_f), maximum(Nu_data_v)]))
PVlim = (minimum([minimum(PV_data_f), minimum(PV_data_v)]), maximum([maximum(PV_data_f), maximum(PV_data_v)]))

gmax = maximum([maximum(abs, geos_balance_f[2:end]), maximum(abs, geos_balance_v[2:end])])
dmax = maximum([maximum(abs, diff_balance_f), maximum(abs, diff_balance_v)])
smax = maximum([maximum(abs, stat_balance_f), maximum(abs, stat_balance_v)])
PVmax = maximum([maximum(abs, PV_data_f), maximum(abs, PV_data_v)])

glim = (-gmax, gmax)
dlim = (-dmax, dmax)
slim = (-smax, smax)
tlim = (0, maximum([maximum(v_data_f.times), maximum(v_data_v.times)]))

heatmap!(axb_f, xC_f, zC_f, bn_f, colormap=Reverse(:RdBu_10), colorrange=blim_f)
heatmap!(axb_v, xC_v, zC_v, bn_v, colormap=Reverse(:RdBu_10), colorrange=blim_v)

hm_geos = heatmap!(axgeos_f, xC_f, zC_f, geosn_f, colormap=Reverse(:RdBu_10), colorrange=glim)
heatmap!(axgeos_v, xC_v, zC_v, geosn_v, colormap=Reverse(:RdBu_10), colorrange=glim)

Colorbar(fig[1:2, 5], hm_geos)

hm_diff = heatmap!(axdiff_f, xC_v, zC_v, diffn_f, colormap=Reverse(:RdBu_10), colorrange=dlim)
heatmap!(axdiff_v, xC_v, zC_v, diffn_v, colormap=Reverse(:RdBu_10), colorrange=dlim)

Colorbar(fig[1:2, 8], hm_diff)

hm_stat = heatmap!(axstat_f, xC_v, zC_v, statn_f, colormap=Reverse(:RdBu_10), colorrange=slim)
heatmap!(axstat_v, xC_v, zC_v, statn_v, colormap=Reverse(:RdBu_10), colorrange=slim)

Colorbar(fig[1:2, 11], hm_stat)

hm_PV = heatmap!(axPV_f, xC_f, zC_f, PVn_f, colorrange=PVlim)
heatmap!(axPV_v, xC_v, zC_v, PVn_v, colorrange=PVlim)

Colorbar(fig[3:4, 5], hm_PV)

lines!(axB, Bn_f, zC_f, label="Flux BC")
lines!(axB, Bn_v, zC_v, label="Value BC")

lines!(axNu, Nun_f, zF_f, label="Flux BC")
lines!(axNu, Nun_v, zF_v, label="Value BC")

xlims!(axB, Blim)
xlims!(axNu, Nulim)

axislegend(axB, position=:lb)
axislegend(axNu)

lines!(axr, v_data_f.times[2:end], geos_balance_ratio_rms_f[2:end], label="Flux")
lines!(axr, v_data_v.times[2:end], geos_balance_ratio_rms_v[2:end], label="Value")

# lines!(axg, v_data_f.times, geos_balance_rms_f[:], label="ϕ = fv - ∂x(p), Flux BC")
# lines!(axg, v_data_f.times, diff_balance_rms_f[:], label="ϕ = ν∇²u, Flux BC")

# lines!(axg, v_data_v.times, geos_balance_rms_v[:], label="ϕ = fv - ∂x(p), Value BC")
# lines!(axg, v_data_v.times, diff_balance_rms_v[:], label="ϕ = ν∇²u, Value BC")

axislegend(axr, position=:lt)
vlines!(axr, t)
xlims!(axr, tlim)
# ylims!(axr, glim)

lines!(axw, w_data_f.times[2:end], w_rms_f[2:end], label="Flux")
lines!(axw, w_data_v.times[2:end], w_rms_v[2:end], label="Value")
axislegend(axw, position=:lt)
vlines!(axw, t)
xlims!(axw, tlim)

record(fig, "Data/Ta1.0e6_Nz_128_fv_comparison_test_w_PV_10.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end

@info "Animation complete"
##
# record(fig, "$(FILE_DIR)/rayleighbenard_convection.mp4", 1:300, framerate=10) do nn
#     n[] = nn
# end
