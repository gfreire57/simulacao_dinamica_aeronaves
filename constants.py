''' Dados da aeronave Mirage e constantes '''

# Dados de Peso, C.G., inércias, e geometria da Aeronave Hospedeiro 
S = 36 #area m2
c = 5.25 # corda m
m = 7400 # massa kg
b = 8.9 # envergadura m 

Iyz = 0
Ixy = 0
Ixx = 0.9*10**4 # kg*m2
Iyy = 5.4*10**4 # kg*m2
Izz = 6.0*10**4 # kg*m2
Ixz = 1.8*10**3 # kg*m2

# Derivadas de estabilidade da Aeronave Hospedeiro
C_L_alpha   = 2.204 #rad^-1
C_L_delta_p = 0.7 #rad^-1
C_L_q       = 0
C_D_0       = 0.015
k           = 0.4
C_m_0       = 0
C_m_q       = -0.4
C_m_delta_p = -0.45
C_m_alpha   = -0.17
C_y_beta    = -0.6
C_y_delta_a = 0.01
C_y_delta_r = 0.075
C_y_delta_l = 0.075
C_r_beta    = -0.05
C_r_p       = -0.25
C_r_r       = 0.06
C_r_delta_a = -0.3
C_r_delta_r = 0.019
C_n_beta    = 0.150
C_n_p       = 0.055
C_n_r       = -0.7
C_n_delta_a = 0
C_n_delta_r = -0.085

# Constantes universais
g = 9.81 # m/s^2
rho_e = 1.225 # kg/m^3


# As constantes c1,c2,c3,c4,c5,c6,c7,c8 e c9 são função das inércias da aeronave, 
# e seus valores encontrados com as equações mostradas abaixo.
# Equações da dinâmica não linear da aeronave 
gamma = (Ixx * Izz) - Ixz**2
c1 = (Iyy - Izz - Ixz**2)/gamma
c2 = (Ixx - Iyy + Izz)*Ixz/gamma
c3 = (Izz)/gamma
c4 = (Ixz)/gamma
c5 = (Izz - Ixx)/Iyy
c6 = Ixz/Iyy
c7 = 1/Iyy
c8 = (Ixx * (Ixx - Iyy)+Ixz**2)/gamma
c9 = (Ixx)/gamma


# LABELS PARA MONTAGEM DOS TÍTULOS DA IMAGEM E DO ARQUIVO PNG GERADO
labels = ['DVt','Dalpha','Dbeta','Dp','Dq','Dr','Dpsi','Dtheta','Dphi','Dh']
