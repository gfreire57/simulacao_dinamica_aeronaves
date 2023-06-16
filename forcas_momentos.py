import numpy as np
from constants import (
    S, b, c
)

# Os parâmetros Fx, Fy e Fz são a soma das forças aerodinâmicas e 
# forças propulsivas atuantes na direção dos eixos Ox (longitudinal), 
# Oy (lateral) e Oz (transversal) da aeronave, respectivamente e M, N. 
# e Lr são os momentos aerodinâmicos de arfagem, de guinada e de rolamento 
# atuantes na aeronave.

def forcas_momentos(T, alpha_f, rho, V_t, CX, CY, CZ, Cm, Cn, Cr):
    # forcas long, lat, transv
    Fx = T*np.cos(alpha_f) + 0.5 * rho * V_t**2 * CX * S
    Fz = T*np.sin(alpha_f) + 0.5 * rho * V_t**2 * CZ * S
    Fy = 0.5 * rho * V_t**2 * CY * S
    # momentos: arfage, guinada, rolagem
    M = 0.5 * rho * V_t**2 * c * Cm * S
    N = 0.5 * rho * V_t**2 * b * Cn * S
    Lr = 0.5 * rho * V_t**2 * b * Cr * S
    return [Fx, Fy, Fz, M, N, Lr]