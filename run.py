# %matplotlib qt

from classDinamica import *
condicoes_voo = [
    1_000, # H
    200,  # Vt
    0,  # beta
    0 # psip
]
# tempo_voo = 0.5
tempo_voo = 300

X0=[
    0,  # delta_profundor
    0,  # delta_aileron  
    0,  # delta_leme     
    0.50,  # pi_comando     
    0.05,  # alpha          
    0.1,  # theta          
    0   # phi            
]

perturbacao = [
    10, #DVt
    0, #Dalpha
    0, #Dbeta
    0, #Dp
    0, #Dq
    0, #Dr
    0, #Dpsi
    0, #Dtheta
    0, #Dphi
    0, #Dh
]
simulacao = simDinamica(condicoes_voo, tempo_voo)

# OPÇÃO DE RODAR DIRETAMENTE
simulacao.runSimulation(
    X0_equilibrio             = X0,
    perturbacao               = perturbacao
)

# # OPÇÃO DE RODAR MODULO POR MODULO
# valEquilibrio = simulacao.calcEquilibrio(X0=X0)
# resposta_dinamica = simulacao.calcDinamica(resultadosEquilibrio=valEquilibrio, perturbacao=perturbacao)
# simulacao.plot_results(resposta_dinamica)

