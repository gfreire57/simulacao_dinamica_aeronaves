# simulacao_dinamica_aeronaves
Programa de simulação dinâmica simples de aeronaves. Permite analisar o comportamento variável da aeronave sob perturbações, como doublet, rajadas, rolamento puro, degrau, etc.

Autor: Gabriel Hasmann Freire Moraes
2023-06


# Manual de uso
Classe simDinamica aceita 2 parametros iniciais: condições de voo (vetor) e tempo de voo (inteiro).
    
    * Condições de voo: [He, V_t, beta, psiponto] (valores em m, m/s e graus)
    * tempo_voo: inteiro, ex: 300 segundos
Chute inicial para o vetor x0 (condição de voo inputada no equilíbrio), dá-se como argumentos, na ordem:
    
    * delta_profundor (°)
    * delta_aileron (°)
    * delta_leme (°)
    * pi_comando (% de potência do motor)
    * alpha (°)
    * theta (°)
    * phi (°)

## SIMULAÇÃO DE DOUBLET
Para simulação de doublet, mais 3 itens devem ser passados para simDinamica:
    
    * vet_tempo: vetor com dados de tempo
    * doublet: vetor com dados de posição da superfície/potência do motor em graus e em %, respectivamente.
    * superfície_doublet: superfície perturbada: 'profundor', 'aileron', 'leme', 'motor'.
    OBS 1: os vetores vet_tempo e doublet devem ser do mesmo tamanho (ver exemplo abaixo)
    OBS 2: Tempo de voo deve ser igual ao ultimo item de vet_tempo caso deseja-se rodar simulação de doublet

Exemplo de simulação com doublet de profundor:
    
    vet_tempo     = [0, 0.9, 1.0, 1.9, 2.0, 2.9, 3.0, 10]
    doublet       = [0,   0,   1,   1,  -1,  -1,   0,  0]
    superficie_doublet = 'profundor'

Profundor terá perturbação de +1° entre 1 e 2 segundos e depois
    de -1° de 2 a 3 segundos, retornando à zero depois de 3s.
