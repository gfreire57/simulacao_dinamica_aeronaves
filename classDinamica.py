# projeto_controle_mirage.ipynb
import os
import numpy as np
from utils import *
from constants import *
from scipy import optimize
from scipy.integrate import solve_ivp
# Modelo do motor da anv hospedeiro
from motor import T_motor_anv
from forcas_momentos import forcas_momentos

'''
   Programa de simulação dinâmica do laboratório de dinâmica de voo (EMA017).
   Autor: Gabriel Hasmann Freire Moraes
'''

# classe
class simDinamica():
    print('''Classe simDinamica() aceita 2 parametros iniciais: condições de voo (lista) e tempo de voo (inteiro).
    * Condições de voo: [He, V_t, beta, psiponto] (valores em m, m/s e graus)
    * tempo_voo: inteiro, ex: 300 segundos
    ''')
    def __init__(self, condicoesVoo: list, tempo_voo) -> None:
        # condicoes iniciais
        self.He          = condicoesVoo[0]
        self.V_t         = condicoesVoo[1]
        self.beta        = condicoesVoo[2]/57.3 # de grau para rad
        self.psiponto    = condicoesVoo[3]/57.3 # de grau para rad
        self.tempo_voo = tempo_voo

    def runSimulation(self, X0_equilibrio, perturbacao):
        ''' Função que roda toda a simulação '''

        # Minimiza funcao equilibrio e acha condições de voo sem pertubação
        valEquilibrio = self.calcEquilibrio(X0=X0_equilibrio)

        resposta_dinamica = self.calcDinamica(
            resultadosEquilibrio=valEquilibrio, 
            perturbacao=perturbacao
        )
        
        self.plot_results(results=resposta_dinamica, perturbacao=perturbacao)

    def calcEquilibrio(self, X0):
        print("Rodando minimização da função de equilíbrio...")
        min_values = optimize.minimize(self.equilibrio, X0, tol=1e-20) # sao 7 iniciais

        print(f'''      Valores para equilíbrio:
                delta_profundor: {min_values.x[0]:.3f} rad
                delta_aileron: {min_values.x[1]:.3f} rad
                delta_leme: {min_values.x[2]:.3f} rad
                pi_comando: {min_values.x[3]:.3f} rad
                alpha: {min_values.x[4]:.3f} rad
                theta: {min_values.x[5]:.3f} rad
                phi: {min_values.x[6]:.3f} rad''')
        # print(min_values)
        print(f"      f_min: {min_values.fun}")
        return min_values

    def calcDinamica(self, resultadosEquilibrio, perturbacao):

        self.delta_profundor   = resultadosEquilibrio.x[0]
        self.delta_aileron     = resultadosEquilibrio.x[1]
        self.delta_leme        = resultadosEquilibrio.x[2]
        self.pi_comando        = resultadosEquilibrio.x[3]
        alpha     = resultadosEquilibrio.x[4]
        theta     = resultadosEquilibrio.x[5]
        phi       = resultadosEquilibrio.x[6]

        print("Rodando cálculo da função de dinâmica...")

        p = -self.psiponto*np.sin(theta)
        q = self.psiponto*np.cos(theta)*np.sin(phi)
        r = self.psiponto*np.cos(theta)*np.cos(phi)

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
        t = list(np.arange(0, self.tempo_voo, 0.1, dtype=float))

        sol = solve_ivp(self.dinamica, [0, self.tempo_voo], y0, t_eval=t)
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
        
        _, _, rho = atmPadrao(h=He_variable)
        alpha_linha = 0 # ?
        alpha_f = 0
        C_L_alpha_linha = 0 # ? Considera como 0 mesmo para simplificar (relevante para vibrações, movs rapidos)
        C_L_0 = 0 # ?

        # print(f"delta_leme: {self.delta_leme} /delta_profundor: {self.delta_profundor} /delta_aileron: {self.delta_aileron}")
        
        CL = C_L_0 + (C_L_alpha * alpha) + (C_L_delta_p * self.delta_profundor) + (C_L_q * (q * c)/V_t)# + (C_L_alpha_linha * (alpha_linha * c)/V_t)
        CD = C_D_0 + k * CL**2
        CYa = (C_y_beta * beta) + (C_y_delta_a * self.delta_aileron)  + (C_y_delta_r * self.delta_leme)
        
        CX = -(np.cos(alpha) * np.cos(beta) * CD) - (np.cos(alpha) * np.sin(beta) * CYa) + (np.sin(alpha) * CL)
        CY = -np.sin(beta) * CD + np.cos(beta) * CYa
        CZ = (-np.sin(alpha) * np.cos(beta) * CD) - (np.sin(alpha) * np.sin(beta) * CYa) - (np.cos(alpha) * CL)
        
        Cm = C_m_0 + (C_m_alpha*alpha) + (C_m_delta_p*self.delta_profundor) + C_m_q*(q*c)/V_t
        Cn = (C_n_beta*beta) + (C_n_delta_r*self.delta_leme) + (C_n_delta_a*self.delta_aileron) + C_n_r*(r*b)/V_t + C_n_p*(p*b)/V_t
        Cr = (C_r_beta*beta) + (C_r_delta_r*self.delta_leme) + (C_r_delta_a*self.delta_aileron) + C_r_p*(p*b)/V_t + C_r_r*(r*b)/V_t
        
        T = T_motor_anv(V_t, self.pi_comando, rho)

        Fx, Fy, Fz, M, N, Lr = forcas_momentos(T, alpha_f, rho, V_t, CX, CY, CZ, Cm, Cn, Cr)
        
        # Vtotal decomposta em 3 componentes
        u = V_t * np.cos (alpha) * np.cos (beta)
        v = V_t * np.sin (beta)
        w = V_t * np.sin (alpha) * np.cos (beta)
        # derivadas das velocidades
        u_linha = r*v - q*w - g*np.sin(theta) + Fx/m
        v_linha = -r*u + p*w + g*np.sin(phi)*np.cos(theta) + Fy/m
        w_linha = q*u - p*v+ g*np.cos(phi)*np.cos(theta) + Fz/m

        phi_linha = p + np.tan(theta)*(q*np.sin(phi) + r*np.cos(phi)) # correto
        theta_linha = q*np.cos(phi) - r*np.sin(phi)
        psi_linha = (q*np.sin(phi) + r*np.cos(phi))/np.cos(theta) # correto

        P_linha = (c1 * r + c2*p)*q + c3*Lr + c4*N
        Q_linha = c5* p * r - c6*(p**2 - r**2) + c7*M
        R_linha = (c8*p - c2*r)*q + c4*Lr + c9*N # correto

        h_linha = u*np.sin(theta) - v*np.sin(phi)*np.sin(theta) - w*np.cos(phi)*np.cos(theta)

        V_t_linha = (u*u_linha + v*v_linha + w*w_linha)/V_t

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

        import matplotlib.pyplot as plt
        # plt.style.use('fivethirtyeight')

        fontdict = {
            'fontsize': 9,
            'fontweight': 'semibold',
            'color': '#000000',
            'verticalalignment': 'baseline',
            'fontfamily': 'Arial'
            # 'horizontalalignment': loc 
        }

        fig = plt.figure(figsize=(10,8), layout="constrained")
        titulo, dict_perturbacao = monta_titulo(perturbacao=perturbacao)
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
        phi_ax.set_title("Phi x t", fontdict=fontdict)
        theta_ax.plot(t_interval, theta_linha)
        theta_ax.set_title("Theta x t", fontdict=fontdict)
        psi_ax.plot(t_interval, psi_linha)
        psi_ax.set_title("Psi x t", fontdict=fontdict)
        rolagem_ax.plot(t_interval, P_linha)
        rolagem_ax.set_title("Rolagem (P)x t", fontdict=fontdict)
        arf_ax.plot(t_interval, Q_linha)
        arf_ax.set_title("Arfagem (Q) x t", fontdict=fontdict)
        guinada_ax.plot(t_interval, R_linha)
        guinada_ax.set_title("Guinada (R) x t", fontdict=fontdict)
        H_ax.plot(t_interval, h_linha)
        H_ax.set_title("Altitude (H) x t", fontdict=fontdict)
        Vt_ax.plot(t_interval, V_t_linha)
        Vt_ax.set_title("Velocidade (Vt) x t", fontdict=fontdict)
        alpha_ax.plot(t_interval, alpha_linha)
        alpha_ax.set_title("Alpha x t", fontdict=fontdict)
        beta_ax.plot(t_interval, beta_linha)
        beta_ax.set_title("Beta x t", fontdict=fontdict)

        def annotate_axes(fig):
            for ax in fig.axes:
                # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center", color='#000000', fontsize=6)
                # ax.set_xticks(t1)
                ax.grid(True, linestyle='-.')
                ax.tick_params(labelbottom=True, labelleft=True, labelsize=7, labelcolor='#000000')

        annotate_axes(fig)

        # Velocidade
        # ax2 = plt.subplot(223)
        # ax2.grid(visible=True)
        # # ax2.plot(t_interval, V_t_linha)
        # ax2.plot(t1, f2(t1))
        # ax2.set_title("Velocidade x Tempo")

        # plt.show()
        if not os.path.exists('results'):
            # If it doesn't exist, create it
            os.makedirs('results')
        image_path = os.path.join(os.getcwd(), 'results')
        fig_title = "simDim__" + '_'.join({f"{k}{v}" for k, v in dict_perturbacao.items() if v != 0 }) + ".png"
        save_path = os.path.join(image_path, fig_title)
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico salvo na pasta {save_path}")