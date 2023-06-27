# utils.py
import numpy as np

def atmPadrao(h: float, unidade: str = 'metro'):
    ''' Cálculo de atm padrão dada a altitude (em pés ou metros) '''

    if unidade == 'metro': h = h * 0.3048
    
    # nivel mar
    p0 = 1013.25 # pressao, kPa
    rho0 = 1.225 # densidade ar, kg/m^3
    T0 = 288.15 # temp ar, K
    R = 286.9 # const universal, J/Kg*K
    g = 9.81 # m/s^2
    alfa_l = -0.0065
    g_alfa_R = g/(alfa_l*R)
    # print("altura h: ", h)
    if (0 <= h <= 11_000):
       Temp = T0 + alfa_l*h
       Pressao = p0*(Temp/T0)**(-g/(alfa_l*R))
       rho = rho0 *(Temp/T0)**(-g_alfa_R-1)
    else: #h > 11000
        _, P_11k, rho_11k = atmPadrao(11000) # pressao aos 11k pés
        Temp = 216.63
        Pressao = P_11k*np.exp(-g*(h-11000)/(R*Temp))
        rho = rho_11k*np.exp(-g*(h-11000)/(R*Temp))
    # else:
    #     raise Exception("Altura inválida: ", h)

    return Temp, Pressao, rho

def monta_titulo(perturbacao):
    ''' Função que monta título do plot final '''
    labels = ['DVt','Dalpha','Dbeta','Dp','Dq','Dr','Dpsi','Dtheta','Dphi','Dh']
    dict_perturbacao = dict(zip(labels, perturbacao))
    # print(dict_perturbacao)

    title = "Resultado da simulação ("
    title = title + ', '.join({f"{k} = {v}" for k, v in dict_perturbacao.items() if v != 0 }) + ')'

    return title, dict_perturbacao
