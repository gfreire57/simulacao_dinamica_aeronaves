# projeto_controle_mirage.ipynb
import os
import numpy as np
from utils import *
from constants import *
from scipy import optimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
# Modelo do motor da anv hospedeiro
from motor import T_motor_anv
from forcas_momentos import forcas_momentos
import matplotlib.pyplot as plt

'''
   Programa de simulação dinâmica do laboratório de dinâmica de voo (EMA017).
   Autor: Gabriel Hasmann Freire Moraes
'''

# classe
class simDinamica():
    '''
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

    ---SIMULAÇÃO DE DOUBLET---
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
    '''
    print("Leia o arquivo ReadMe antes de usar a classe! Lá estão as instruções básicas.")
    def __init__(
            self, 
            condicoesVoo          : list,
            tempo_voo             : int,
            vet_tempo             : list[float] = [],
            doublet               : list[float] = [], # em graus
            superficie_doublet    : str = '', # 'profundor', 'aileron', 'leme', 'tracao'
    ): 
        # condicoes iniciais
        self.He          = condicoesVoo[0]
        self.V_t         = condicoesVoo[1]
        self.beta        = condicoesVoo[2]/57.3 # de grau para rad
        self.psiponto    = condicoesVoo[3]/57.3 # de grau para rad
        self.tempo_voo   = tempo_voo

        self.is_doublet  = False
        self.superficie_doublet = ''

        if bool(vet_tempo) and bool(doublet):
            # Verifica se num elementos nas duas listas são iguais e se todos os valores são float
            if (len(vet_tempo) == len(doublet)):# and (all(isinstance(x, float) for x in vet_tempo)) and (all(isinstance(x, float) for x in doublet)):
                self.is_doublet   = True
                self.vet_tempo    = vet_tempo
                self.doublet      = doublet # conversão para rad feita em calcDinamica
                self.superficie_doublet = superficie_doublet
            else                  : 
                raise Exception("Há algum problema com os vetores vet_tempo e doublet. Verifique.")

    def runSimulation(self, X0_equilibrio, perturbacao):
        ''' Função que roda toda a simulação '''

        # Minimiza funcao equilibrio e acha condições de voo sem pertubação
        valEquilibrio = self.calcEquilibrio(X0=X0_equilibrio)

        resposta_dinamica = self.calcDinamica(
            resultadosEquilibrio=valEquilibrio, 
            perturbacao=perturbacao, 
        )
        
        self.plot_results(results=resposta_dinamica, perturbacao=perturbacao)

    def calcEquilibrio(self, X0):
        print("Rodando minimização da função de equilíbrio...")
        min_values = optimize.minimize(self.equilibrio, X0, tol=1e-20) # sao 7 iniciais

        print(f'''      Valores encontrados para voo em equilíbrio:
                delta_profundor: {min_values.x[0]:.3f} rad
                delta_aileron: {min_values.x[1]:.3f} rad
                delta_leme: {min_values.x[2]:.3f} rad
                pi_comando: {min_values.x[3]:.3f} %
                alpha: {min_values.x[4]:.3f} rad
                theta: {min_values.x[5]:.3f} rad
                phi: {min_values.x[6]:.3f} rad''')
        # print(min_values)
        print(f"      f_min: {min_values.fun}")
        return min_values

    def calcDinamica(self, resultadosEquilibrio, perturbacao):

        self.delta_profundor = resultadosEquilibrio.x[0]
        self.delta_aileron   = resultadosEquilibrio.x[1]
        self.delta_leme      = resultadosEquilibrio.x[2]
        self.pi_comando      = resultadosEquilibrio.x[3]
        alpha     = resultadosEquilibrio.x[4]
        theta     = resultadosEquilibrio.x[5]
        phi       = resultadosEquilibrio.x[6]

        print("Rodando cálculo da função de dinâmica...")

        p = -self.psiponto*np.sin(theta)
        q = self.psiponto*np.cos(theta)*np.sin(phi)
        r = self.psiponto*np.cos(theta)*np.cos(phi)
        
        # Acréscimo das perturbações
        y0 = [
            self.V_t   + perturbacao[0], #DVt,
            alpha      + perturbacao[1]/57.3, #Dalpha,
            self.beta  + perturbacao[2]/57.3, #Dbeta,
            p          + perturbacao[3]/57.3, #Dp,
            q          + perturbacao[4]/57.3, #Dq,    
            r          + perturbacao[5]/57.3, #Dr,
            0          + perturbacao[6]/57.3, #Dpsi, # psi = 0
            theta      + perturbacao[7]/57.3, #Dtheta,
            phi        + perturbacao[8]/57.3, #Dphi,
            self.He    + perturbacao[9]  #Dh
        ]
 
        t_span = [0, self.tempo_voo]
        t_eval = list(np.arange(0, self.tempo_voo, 0.01, dtype=float)) # precisa ter o mesmo num de elementso que doublet
        
        # no caso de doublet, criamos um objeto interp1d que irá devolver, na func 
        # dinamica, o valor da perturbação no tempo t e irá incrementar esse valor 
        # ao valor inicial em equilíbrio ao longo da simulação dinâmica.
        if self.is_doublet:
            if self.superficie_doublet != "tracao":
                self.doublet = np.divide(self.doublet, 57.3) # conversão graus para rad
            self.doublet_interp = interp1d(x=self.vet_tempo, y=self.doublet) # objeto interp1d

            # plota perturbação
            xnew = np.arange(0, 0.1, 10)
            ynew = self.doublet_interp(xnew)
            plt.plot(self.vet_tempo, self.doublet, '-', xnew, ynew, '-')
            plt.show()

        sol = solve_ivp(self.dinamica, t_span=t_span, y0=y0, t_eval=t_eval)

        if sol.success: print(f"    Solução encontrada")
        else: raise Exception("     !> Houve um problema na simulação.")

        return sol

    def equilibrio(self, X0):
        delta_profundor   = X0[0]
        delta_aileron     = X0[1]
        delta_leme        = X0[2]
        pi_comando        = X0[3]
        alpha             = X0[4]
        theta             = X0[5]
        phi               = X0[6]

        # velocidades angulares
        p = -self.psiponto*np.sin(theta)
        q = self.psiponto*np.cos(theta)*np.sin(phi)
        r = self.psiponto*np.cos(theta)*np.cos(phi)

        _, _, rho = atmPadrao(h=self.He)
        
        # ALpha_f => 
        alpha_f = 0
        alpha_linha = 0 # ?
        C_L_alpha_linha = 0 # ? Considera como 0 mesmo para simplificar (relevante para vibrações, movs rapidos)
        C_L_0 = 0 # ?
        
        CL = C_L_0 + (C_L_alpha * alpha) + (C_L_delta_p * delta_profundor) + (C_L_q * (q * c)/self.V_t)# + (C_L_alpha_linha * (alpha_linha * c)/self.V_t)
        CD = C_D_0 + k * CL**2
        CYa = (C_y_beta * self.beta) + (C_y_delta_a * delta_aileron)  + (C_y_delta_r * delta_leme)
        
        CX = -(np.cos(alpha) * np.cos(self.beta) * CD) - (np.cos(alpha) * np.sin(self.beta) * CYa) + (np.sin(alpha) * CL)
        CY = -(np.sin(self.beta) * CD) + (np.cos(self.beta) * CYa)
        CZ = -(np.sin(alpha) * np.cos(self.beta) * CD) - (np.sin(alpha) * np.sin(self.beta) * CYa) - (np.cos(alpha) * CL)
        
        Cm = C_m_0 + (C_m_alpha*alpha) + (C_m_delta_p*delta_profundor) + C_m_q*(q*c)/self.V_t
        Cn = (C_n_beta*self.beta) + (C_n_delta_r*delta_leme) + (C_n_delta_a*delta_aileron) + C_n_r*(r*b)/self.V_t + C_n_p*(p*b)/self.V_t
        Cr = (C_r_beta*self.beta) + (C_r_delta_r*delta_leme) + (C_r_delta_a*delta_aileron) + C_r_p*(p*b)/self.V_t + C_r_r*(r*b)/self.V_t
        
        T = T_motor_anv(self.V_t, pi_comando, rho)

        Fx, Fy, Fz, M, N, Lr = forcas_momentos(T, alpha_f, rho, self.V_t, CX, CY, CZ, Cm, Cn, Cr)
        
        # Vtotal decomposta em 3 componentes
        u = self.V_t * np.cos (alpha) * np.cos (self.beta)
        v = self.V_t * np.sin (self.beta)
        w = self.V_t * np.sin (alpha) * np.cos (self.beta)
        # derivadas das velocidades
        u_linha = r*v - q*w - g*np.sin(theta) + Fx/m
        v_linha = -r*u + p*w + g*np.sin(phi)*np.cos(theta) + Fy/m
        w_linha = q*u - p*v+ g*np.cos(phi)*np.cos(theta) + Fz/m

        # phi_linha = p + tan(theta)*(q*np.sin(phi) + r*cos(phi))
        # theta_linha = q*cos(phi) - r*np.sin(phi)
        # psi_linha = (q*np.sin(phi) + r*cos(phi))/cos(theta)

        P_linha = (c1 * r + c2*p)*q + c3*Lr + c4*N
        Q_linha = c5* p * r - c6*(p**2 - r**2) + c7*M
        R_linha = (c8*p - c2*r)*q + c4*Lr + c9*N # correto

        h_linha = u*np.sin(theta) - v*np.sin(phi)*np.sin(theta) - w*np.cos(phi)*np.cos(theta)

        # V_t_linha = (u*u_linha + self.V_t*v_linha +w*w_linha)/self.V_t
        # alpha_linha = (u*w_linha - w*u_linha)/(u**2 + w**2)
        # beta_linha = (v_linha*self.V_t - self.V_t*V_t_linha)/(cos(beta)*self.V_t**2)

        # return [u_linha, v_linha, w_linha, P_linha, Q_linha, R_linha, h_linha]
        return u_linha**2 + v_linha**2 +  w_linha**2 +  P_linha**2 +  Q_linha**2 +  R_linha**2 +  h_linha**2
    
    def dinamica(self, t, y0):

        V_t           = y0[0]
        alpha         = y0[1]
        beta          = y0[2]
        p             = y0[3]
        q             = y0[4]
        r             = y0[5]
        psi           = y0[6]
        theta         = y0[7]
        phi           = y0[8]
        He_variable   = y0[9]
        
        delta_profundor_DINAMICA    = self.delta_profundor
        delta_aileron_DINAMICA      = self.delta_aileron
        delta_leme_DINAMICA         = self.delta_leme
        pi_comando_DINAMICA         = self.pi_comando
        ######### DOUBLET
        if self.is_doublet:
            if   self.superficie_doublet == 'profundor':
                delta_profundor_DINAMICA = self.delta_profundor + self.doublet_interp(t)
            elif self.superficie_doublet == 'aileron':
                delta_aileron_DINAMICA   = self.delta_aileron + self.doublet_interp(t)
            elif self.superficie_doublet == 'leme':
                delta_leme_DINAMICA      = self.delta_leme + self.doublet_interp(t)
            elif self.superficie_doublet == 'tracao':
                pi_comando_DINAMICA      = self.pi_comando + self.doublet_interp(t)
        #########
        _, _, rho = atmPadrao(h=He_variable)
        alpha_linha = 0 # ?
        alpha_f = 0
        C_L_alpha_linha = 0 # ? Considera como 0 mesmo para simplificar (relevante para vibrações, movs rapidos)
        C_L_0 = 0 # ?

        CL = C_L_0 + (C_L_alpha * alpha) + (C_L_delta_p * delta_profundor_DINAMICA) + (C_L_q * (q * c)/V_t)# + (C_L_alpha_linha * (alpha_linha * c)/V_t)
        CD = C_D_0 + k * CL**2
        CYa = (C_y_beta * beta) + (C_y_delta_a * delta_aileron_DINAMICA)  + (C_y_delta_r * delta_leme_DINAMICA)
        
        CX = -(np.cos(alpha) * np.cos(beta) * CD) - (np.cos(alpha) * np.sin(beta) * CYa) + (np.sin(alpha) * CL)
        CY = -np.sin(beta) * CD + np.cos(beta) * CYa
        CZ = (-np.sin(alpha) * np.cos(beta) * CD) - (np.sin(alpha) * np.sin(beta) * CYa) - (np.cos(alpha) * CL)
        
        Cm = C_m_0 + (C_m_alpha*alpha) + (C_m_delta_p*delta_profundor_DINAMICA) + C_m_q*(q*c)/V_t
        Cn = (C_n_beta*beta) + (C_n_delta_r*delta_leme_DINAMICA) + (C_n_delta_a*delta_aileron_DINAMICA) + C_n_r*(r*b)/V_t + C_n_p*(p*b)/V_t
        Cr = (C_r_beta*beta) + (C_r_delta_r*delta_leme_DINAMICA) + (C_r_delta_a*delta_aileron_DINAMICA) + C_r_p*(p*b)/V_t + C_r_r*(r*b)/V_t
        
        T = T_motor_anv(V_t, pi_comando_DINAMICA, rho)

        Fx, Fy, Fz, M, N, Lr = forcas_momentos(T, alpha_f, rho, V_t, CX, CY, CZ, Cm, Cn, Cr)
        
        # Vtotal decomposta em 3 componentes
        u = V_t * np.cos (alpha) * np.cos (beta)
        v = V_t * np.sin (beta)
        w = V_t * np.sin (alpha) * np.cos (beta)

        # derivadas das componentes da velocidade
        u_linha = r*v - q*w - g*np.sin(theta) + Fx/m
        v_linha = -r*u + p*w + g*np.sin(phi)*np.cos(theta) + Fy/m
        w_linha = q*u - p*v+ g*np.cos(phi)*np.cos(theta) + Fz/m
        
        # derivadas dos angulos de orientação da anv em relação ao referencial da terra
        phi_linha = p + np.tan(theta)*(q*np.sin(phi) + r*np.cos(phi)) # correto
        theta_linha = q*np.cos(phi) - r*np.sin(phi)
        psi_linha = (q*np.sin(phi) + r*np.cos(phi))/np.cos(theta) # correto
        
        # Taxas de rolagem (P), arfagem(Q) e guinada (R)
        P_linha = (c1 * r + c2*p)*q + c3*Lr + c4*N
        Q_linha = c5* p * r - c6*(p**2 - r**2) + c7*M
        R_linha = (c8*p - c2*r)*q + c4*Lr + c9*N # correto
        
        # Taxa da altitude
        h_linha = u*np.sin(theta) - v*np.sin(phi)*np.sin(theta) - w*np.cos(phi)*np.cos(theta)
        
        # taxa da velocidade
        V_t_linha = (u*u_linha + v*v_linha + w*w_linha)/V_t
        
        # Taxas da AoA e derrapagem
        alpha_linha = (u*w_linha - w*u_linha)/(u**2 + w**2)
        beta_linha = (v_linha*V_t - v*V_t_linha)/(np.cos(beta)*V_t**2)
        
        # print(f'''V_t_linha {V_t_linha:.2f} /alpha_linha {alpha_linha:.2f} /beta_linha {beta_linha:.2f} /P_linha {P_linha:.2f} /Q_linha {Q_linha:.2f} /R_linha {R_linha:.2f} /psi_linha {psi_linha:.2f} /theta_linha {theta_linha:.2f} /phi_linha {phi_linha:.2f} /h_linha {h_linha:.2f}''')
        dY = [
            V_t_linha,
            alpha_linha,
            beta_linha,
            P_linha,
            Q_linha,
            R_linha,
            psi_linha,
            theta_linha,
            phi_linha,
            h_linha
        ]

        return dY

    def plot_results(self, results, perturbacao):
        ''' Plota resultados. '''
        # plt.style.use('fivethirtyeight')
        V_t_linha   = results.y[0]
        alpha_linha = results.y[1]*360/(2*np.pi) # AoA
        beta_linha  = results.y[2]*360/(2*np.pi) # derrapagem
        P_linha     = results.y[3]*360/(2*np.pi) # rolagem
        Q_linha     = results.y[4]*360/(2*np.pi) # arfagem 
        R_linha     = results.y[5]*360/(2*np.pi) # guinada
        psi_linha   = results.y[6]*360/(2*np.pi)
        theta_linha = results.y[7]*360/(2*np.pi)
        phi_linha   = results.y[8]*360/(2*np.pi)
        h_linha     = results.y[9]
        t_interval  = results.t

        fontdict = {
            'fontsize': 9,
            'fontweight': 'semibold',
            'color': '#000000',
            'verticalalignment': 'baseline',
            'fontfamily': 'Arial'
            # 'horizontalalignment': loc 
        }

        fig = plt.figure(figsize=(10,8), layout="constrained")

        titulo = monta_titulo(
            perturbacao=perturbacao, 
            is_doublet=self.is_doublet, 
            superficie_doublet=self.superficie_doublet
        )
        
        plt.suptitle(titulo,
            fontsize='large',
            # loc='left',
            fontweight='bold',
            style='italic',
            family='monospace'
            )

        phi_ax        = plt.subplot2grid((4, 3), (0,0), rowspan=1, colspan=1)
        theta_ax      = plt.subplot2grid((4, 3), (0,1), rowspan=1, colspan=1)
        psi_ax        = plt.subplot2grid((4, 3), (0,2), rowspan=1, colspan=1)
        rolagem_ax    = plt.subplot2grid((4, 3), (1,0), rowspan=1, colspan=1)
        arf_ax        = plt.subplot2grid((4, 3), (1,1), rowspan=1, colspan=1)
        guinada_ax    = plt.subplot2grid((4, 3), (1,2), rowspan=1, colspan=1)
        H_ax          = plt.subplot2grid((4, 3), (2,0), rowspan=1, colspan=3)
        Vt_ax         = plt.subplot2grid((4, 3), (3,0), rowspan=1, colspan=1)
        alpha_ax      = plt.subplot2grid((4, 3), (3,1), rowspan=1, colspan=1)
        beta_ax       = plt.subplot2grid((4, 3), (3,2), rowspan=1, colspan=1)

        # Plota curvas
        phi_ax.plot(t_interval, phi_linha)
        phi_ax.set_title("Phi (ang rolagem, rad) x t", fontdict=fontdict)
        theta_ax.plot(t_interval, theta_linha)
        theta_ax.set_title("Theta (ang arfagem, rad) x t", fontdict=fontdict)
        psi_ax.plot(t_interval, psi_linha)
        psi_ax.set_title("Psi (ang guinada, rad) x t", fontdict=fontdict)
        rolagem_ax.plot(t_interval, P_linha)
        rolagem_ax.set_title("P (taxa rolagem, , rad/s) x t", fontdict=fontdict)
        arf_ax.plot(t_interval, Q_linha)
        arf_ax.set_title("Q (taxa arfagem, rad/s) x t", fontdict=fontdict)
        guinada_ax.plot(t_interval, R_linha)
        guinada_ax.set_title("R (taxa guinada, rad/s) x t", fontdict=fontdict)
        H_ax.plot(t_interval, h_linha)
        H_ax.set_title("Altitude (H, metros) x t", fontdict=fontdict)
        Vt_ax.plot(t_interval, V_t_linha)
        Vt_ax.set_title("Velocidade (Vt, m/s) x t", fontdict=fontdict)
        alpha_ax.plot(t_interval, alpha_linha)
        alpha_ax.set_title("Alpha (rad) x t", fontdict=fontdict)
        beta_ax.plot(t_interval, beta_linha)
        beta_ax.set_title("Beta (rad) x t", fontdict=fontdict)

        def annotate_axes(fig):
            for ax in fig.axes:
                # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center", color='#000000', fontsize=6)
                # ax.set_xticks(t1)
                ax.grid(True, linestyle='-.')
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=7, labelcolor='#000000')

        annotate_axes(fig)

        if not os.path.exists('results'):
            # If it doesn't exist, create it
            os.makedirs('results')
        image_path = os.path.join(os.getcwd(), 'results')

        if self.is_doublet:
            fig_title = "simDim__doublet_"+ self.superficie_doublet + ".png"
        else:
            dict_perturbacao = dicionarioValoresPerturbacao(perturbacao=perturbacao)
            fig_title = "simDim__" + '_'.join({f"{k}{v}" for k, v in dict_perturbacao.items() if v != 0 }) + ".png"

        save_path = os.path.join(image_path, fig_title)
        plt.savefig(save_path, dpi=300)

        print(f"Gráfico salvo na pasta:\n{save_path}")
        plt.show()