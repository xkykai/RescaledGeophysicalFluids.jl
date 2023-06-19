using Oceananigans

FILE_DIR = "Data/2D_no_wind_uv_tfbf_b_tfbf_Ra_1.2e7_Ta_1.0e9_alpha_1.0_Nz_128"
B_data = FieldTimeSeries("$(FILE_DIR)/averaged_timeseries.jld2", "B")

Ra = B_data[1, 1, 1, end] - B_data[1, 1, end, end]
@info "Effective Ra = $(Ra)"