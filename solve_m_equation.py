#%%
from sympy import *
import numpy as np

#%%
m_sym, k_sym, Ta_sym, Ra_sym, alpha_sym, Pr_sym, nu_sym, H_sym, rho_0_sym, z_sym = symbols("m k Ta Ra alpha Pr nu H, rho_0 z")

nu = 1
Pr = 1
H = 1
Ra = 1000
Ta = 1000
k = 4

kappa_sym = nu_sym / Pr_sym
f_sym = sqrt(Ta_sym * nu_sym**2 / H_sym**4)
S_sym = Ra_sym * nu_sym * kappa_sym / H_sym**4

#%%
expr = (m_sym**2 - k_sym**2)**3 + Ta_sym * m_sym**2 + Ra_sym * k_sym**2

ms_sol = solve(expr, m_sym)

# for i in range(len(ms_sol)):
#     ms_sol[i] = ms_sol[i].subs([(nu_sym, nu), (Ta_sym, Ta), (Ra_sym, Ra), (k_sym, k)])

ms_sym = np.array([k_sym, -k_sym, *ms_sol])
#%%
def compute_coefficients(k, m, nu, kappa, f, S, rho_0):
    A = Matrix([[nu*(k**2 - m**2), -f, 0, 0], 
                [f, nu*(k**2 - m**2), 0, 0], 
                [0, 0, nu*(k**2 - m**2), -1], 
                [0, 0, S, kappa*(k**2 - m**2)]])
    
    b = Matrix([-I*k/rho_0, 0, -m/rho_0, 0])

    return A.solve(b)

uvwb_sym = compute_coefficients(k_sym, m_sym, nu_sym, kappa_sym, f_sym, S_sym, rho_0_sym)

uvwbs_sym = [uvwb_sym.subs(m_sym, ms_sym[i]) * exp(ms_sym[i]*z_sym) for i in range(len(ms_sym))]

#%%
u_top_sym = Matrix([ms_sym[i] * uvwbs_sym[i][0].subs(z_sym, H_sym/2) for i in range(len(uvwbs_sym))])
v_top_sym = Matrix([ms_sym[i] * uvwbs_sym[i][1].subs(z_sym, H_sym/2) for i in range(len(uvwbs_sym))])
w_top_sym = Matrix([uvwbs_sym[i][2].subs(z_sym, H_sym/2) for i in range(len(uvwbs_sym))])
b_top_sym = Matrix([uvwbs_sym[i][3].subs(z_sym, H_sym/2) for i in range(len(uvwbs_sym))])

u_bot_sym = Matrix([ms_sym[i] * uvwbs_sym[i][0].subs(z_sym, -H_sym/2) for i in range(len(uvwbs_sym))])
v_bot_sym = Matrix([ms_sym[i] * uvwbs_sym[i][1].subs(z_sym, -H_sym/2) for i in range(len(uvwbs_sym))])
w_bot_sym = Matrix([uvwbs_sym[i][2].subs(z_sym, -H_sym/2) for i in range(len(uvwbs_sym))])
b_bot_sym = Matrix([uvwbs_sym[i][3].subs(z_sym, -H_sym/2) for i in range(len(uvwbs_sym))])

def det_bc_matrix(u_top, v_top, w_top, b_top, u_bot, v_bot, w_bot, b_bot):
   A = Matrix.hstack(u_top, v_top, w_top, b_top, u_bot, v_bot, w_bot, b_bot).T
   return A.det()

A_sym = det_bc_matrix(u_top_sym, v_top_sym, w_top_sym, b_top_sym, u_bot_sym, v_bot_sym, w_bot_sym, b_bot_sym)

# function compute_coefficients(k, ms, ν, κ, f, S, ρ₀)
#     us = [zeros(ComplexF64, 4) for i in 1:length(ms)]
#     for (i, m) in pairs(ms)
#         A = [ν*(k^2 - m^2) -f 0 0;
#             f ν*(k^2 - m^2) 0 0;
#             0 0 ν*(k^2 - m^2) -1;
#             0 0 S κ*(k^2 - m^2);]

#         b = [-im*k/ρ₀, 0, -m/ρ₀, 0]

#         us[i] = A \ b
#     end
#     return us
# end
#%%
u_1_top_sym, u_2_top_sym, u_3_top_sym, u_4_top_sym, u_5_top_sym, u_6_top_sym, u_7_top_sym, u_8_top_sym = symbols("u_1top u_2top u_3top u_4top u_5top u_6top u_7top u_8top")
v_1_top_sym, v_2_top_sym, v_3_top_sym, v_4_top_sym, v_5_top_sym, v_6_top_sym, v_7_top_sym, v_8_top_sym = symbols("v_1top v_2top v_3top v_4top v_5top v_6top v_7top v_8top")
w_1_top_sym, w_2_top_sym, w_3_top_sym, w_4_top_sym, w_5_top_sym, w_6_top_sym, w_7_top_sym, w_8_top_sym = symbols("w_1top w_2top w_3top w_4top w_5top w_6top w_7top w_8top")
b_1_top_sym, b_2_top_sym, b_3_top_sym, b_4_top_sym, b_5_top_sym, b_6_top_sym, b_7_top_sym, b_8_top_sym = symbols("b_1top b_2top b_3top b_4top b_5top b_6top b_7top b_8top")


u_1_bot_sym, u_2_bot_sym, u_3_bot_sym, u_4_bot_sym, u_5_bot_sym, u_6_bot_sym, u_7_bot_sym, u_8_bot_sym = symbols("u_1bot u_2bot u_3bot u_4bot u_5bot u_6bot u_7bot u_8bot")
v_1_bot_sym, v_2_bot_sym, v_3_bot_sym, v_4_bot_sym, v_5_bot_sym, v_6_bot_sym, v_7_bot_sym, v_8_bot_sym = symbols("v_1bot v_2bot v_3bot v_4bot v_5bot v_6bot v_7bot v_8bot")
w_1_bot_sym, w_2_bot_sym, w_3_bot_sym, w_4_bot_sym, w_5_bot_sym, w_6_bot_sym, w_7_bot_sym, w_8_bot_sym = symbols("w_1bot w_2bot w_3bot w_4bot w_5bot w_6bot w_7bot w_8bot")
b_1_bot_sym, b_2_bot_sym, b_3_bot_sym, b_4_bot_sym, b_5_bot_sym, b_6_bot_sym, b_7_bot_sym, b_8_bot_sym = symbols("b_1bot b_2bot b_3bot b_4bot b_5bot b_6bot b_7bot b_8bot")


A = Matrix([[u_1_top_sym, u_2_top_sym, u_3_top_sym, u_4_top_sym, u_5_top_sym, u_6_top_sym, u_7_top_sym, u_8_top_sym], 
            [v_1_top_sym, v_2_top_sym, v_3_top_sym, v_4_top_sym, v_5_top_sym, v_6_top_sym, v_7_top_sym, v_8_top_sym],
            [w_1_top_sym, w_2_top_sym, w_3_top_sym, w_4_top_sym, w_5_top_sym, w_6_top_sym, w_7_top_sym, w_8_top_sym],
            [b_1_top_sym, b_2_top_sym, b_3_top_sym, b_4_top_sym, b_5_top_sym, b_6_top_sym, b_7_top_sym, b_8_top_sym],
            [u_1_bot_sym, u_2_bot_sym, u_3_bot_sym, u_4_bot_sym, u_5_bot_sym, u_6_bot_sym, u_7_bot_sym, u_8_bot_sym],
            [v_1_bot_sym, v_2_bot_sym, v_3_bot_sym, v_4_bot_sym, v_5_bot_sym, v_6_bot_sym, v_7_bot_sym, v_8_bot_sym],
            [w_1_bot_sym, w_2_bot_sym, w_3_bot_sym, w_4_bot_sym, w_5_bot_sym, w_6_bot_sym, w_7_bot_sym, w_8_bot_sym],
            [b_1_bot_sym, b_2_bot_sym, b_3_bot_sym, b_4_bot_sym, b_5_bot_sym, b_6_bot_sym, b_7_bot_sym, b_8_bot_sym]])

det_A = A.det()
#%%