from constants import rho_e

def T_motor_anv(V_t, pi_comando, rho, T_max: float = 40_000, V_e: float = 0.0):
    #  = 40000 # newtons
    eta_ni = 0
    eta_rho = 1
    T = pi_comando * T_max * (rho/rho_e)**eta_rho # (V_t/V_e)**eta_ni *
    return T