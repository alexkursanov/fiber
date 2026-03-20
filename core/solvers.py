"""
Модульные решатели для компонентов модели
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from typing import Optional, Tuple

from config.parameters import ModelParameters
from core.results import IschemiaConfig


class ElectricalSolver:
    """Решатель электрической части для одной клетки"""
    
    def __init__(self, params: ModelParameters, ischemia: IschemiaConfig):
        self.elec = params.elec
        self.ekb = params.ekb
        self.ischemia = ischemia
        self._init_cache()
    
    def _init_cache(self):
        """Инициализация кэша констант"""
        self.sqr_ryr = self.elec.K_RyR ** 2
        self.exp_00001 = np.exp(-0.00001)
        self.f_K = self.elec.f_K
        self.tau_s_ss = self.elec.tau_s_ss
        self.a_eqmin = self.ekb.a_eqmin
        self.s_c = self.ekb.s_c
    
    def rhs(self, time: float, Y: np.ndarray, cell_idx: int, N_elec: float) -> np.ndarray:
        """
        Правая часть ОДУ для одной клетки
        
        Args:
            time: текущее время
            Y: вектор состояния [24]
            cell_idx: индекс клетки (1-based)
            N_elec: N для электрической связи
        
        Returns:
            dYdt: производные [24]
        """
        from models.electrical import calculate_ischemia_params
        
        bzdegree = self.ischemia.get_bzdegree(cell_idx)
        
        ATP_i = self.elec.ATP_i
        K_o = self.elec.K_o
        g_Na = self.elec.g_Na
        J_L = self.elec.J_L
        
        ATP_i, K_o, g_Na, J_L = calculate_ischemia_params(
            self.ischemia.degree, bzdegree, ATP_i, K_o, g_Na, J_L
        )
        
        z_1, z_2, z_3 = Y[0], Y[1], Y[2]
        r1, s, s_slow = Y[3], Y[4], Y[5]
        y = Y[6]
        Ca_SR, Ca_i, K_i, Na_i = Y[7], Y[8], Y[9], Y[10]
        TRPN, V = Y[11], Y[12]
        h, j_sod, m_iNa = Y[17], Y[18], Y[19]
        r_ss, s_ss = Y[20], Y[21]
        
        F = self.elec.F
        R = self.elec.R
        Tem = self.elec.Tem
        V_myo = self.elec.V_myo
        V_myo_uL = self.elec.V_myo_uL
        V_SR_uL = self.elec.V_SR_uL
        
        pi_n1 = self.elec.GeneralTnC * self.s_c * N_elec / (TRPN + 1e-12)
        piv = 1.0 if pi_n1 <= 0.0 else (self.a_eqmin ** pi_n1 if pi_n1 <= 1.0 else self.a_eqmin)
        
        koff = self.elec.C20 * np.exp(-self.elec.qa * TRPN) * piv
        dTRPN = self.elec.BindingConstTnC * (self.elec.GeneralTnC - TRPN) * Ca_i - koff * TRPN
        
        expVL = np.exp((V - self.elec.V_L) / self.elec.del_VL)
        alpha_p = expVL / (self.elec.t_L * (expVL + 1.0))
        
        FVRT = F * V / (R * Tem)
        FVRT_Ca = 2.0 * FVRT
        exp_FVRT_Ca = np.exp(-FVRT_Ca)
        
        if abs(FVRT_Ca) > 1e-9:
            C_oc = ((Ca_i + J_L / self.elec.g_D * self.elec.Ca_o *
                     FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) /
                    (1.0 + J_L / self.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        else:
            C_oc = (Ca_i + J_L / self.elec.g_D * self.elec.Ca_o) / (1.0 + J_L / self.elec.g_D)
        
        sqr_coc = C_oc ** 2
        sqr_y8 = Ca_i ** 2
        
        t_R = self.elec.t_R
        alpha_m = self.elec.alpha_m
        beta_m = self.elec.beta_m
        
        beta_poc = sqr_coc / (t_R * (sqr_coc + self.sqr_ryr))
        beta_pcc = sqr_y8 / (t_R * (sqr_y8 + self.sqr_ryr))
        
        C_co = (Ca_i + self.elec.J_R / self.elec.g_D * Ca_SR) / (1.0 + self.elec.J_R / self.elec.g_D)
        
        epsilon_tmp = (expVL + self.elec.a1) / (self.elec.tau_L * self.elec.K_L * (expVL + 1.0))
        epsilon_pco = C_co * epsilon_tmp
        epsilon_pcc = Ca_i * epsilon_tmp
        epsilon_m = (self.elec.b * (expVL + self.elec.a1) /
                     (self.elec.tau_L * (self.elec.b * expVL + self.elec.a1)))
        
        mu_poc = (sqr_coc + self.elec.c * self.sqr_ryr) / (self.elec.tau_R_1 * (sqr_coc + self.sqr_ryr))
        mu_pcc = (sqr_y8 + self.elec.c * self.sqr_ryr) / (self.elec.tau_R_1 * (sqr_y8 + self.sqr_ryr))
        mu_moc = (self.elec.theta_R * self.elec.TNP_d * (sqr_coc + self.elec.c * self.sqr_ryr) /
                  (self.elec.tau_R_1 * (self.elec.TNP_d * sqr_coc + self.elec.c * self.sqr_ryr)))
        mu_mcc = (self.elec.theta_R * self.elec.TNP_d * (sqr_y8 + self.elec.c * self.sqr_ryr) /
                  (self.elec.tau_R_1 * (self.elec.TNP_d * sqr_y8 + self.elec.c * self.sqr_ryr)))
        
        denom = 1.0 / ((alpha_p + alpha_m) *
                       ((alpha_m + beta_m + beta_poc) * (beta_m + beta_pcc) +
                        alpha_p * (beta_m + beta_poc)))
        
        y_oc = alpha_p * beta_m * (alpha_p + alpha_m + beta_m + beta_pcc) * denom
        y_cc = alpha_m * beta_m * (alpha_m + alpha_p + beta_m + beta_poc) * denom
        
        r_1 = y_oc * mu_poc + y_cc * mu_pcc
        r_2 = (alpha_p * mu_moc + alpha_m * mu_mcc) / (alpha_p + alpha_m)
        r_3 = beta_m * mu_pcc / (beta_m + beta_pcc)
        r_4 = mu_mcc
        
        y_co = alpha_m * (beta_pcc * (alpha_m + beta_m + beta_poc) +
                           beta_poc * alpha_p) * denom
        r_5 = y_co * epsilon_pco + y_cc * epsilon_pcc
        r_6 = epsilon_m
        r_7 = alpha_m * epsilon_pcc / (alpha_p + alpha_m)
        r_8 = epsilon_m
        
        z_4 = 1.0 - z_1 - z_2 - z_3
        
        dz1 = -(r_1 + r_5) * z_1 + r_2 * z_2 + r_6 * z_3
        dz2 = r_1 * z_1 - (r_2 + r_7) * z_2 + r_8 * z_4
        dz3 = r_5 * z_1 - (r_6 + r_3) * z_3 + r_4 * z_4
        
        y_oo = alpha_p * (beta_poc * (alpha_p + beta_m + beta_pcc) +
                            beta_pcc * alpha_m) * denom
        y_ci = alpha_m / (alpha_p + alpha_m)
        y_oi = alpha_p / (alpha_p + alpha_m)
        y_ic = beta_m / (beta_pcc + beta_m)
        y_io = beta_pcc / (beta_pcc + beta_m)
        y_ii = 1.0 - y_oc - y_co - y_oo - y_cc - y_ci - y_ic - y_oi - y_io
        
        J_Rco = self.elec.J_R * (Ca_SR - Ca_i) / (1.0 + self.elec.J_R / self.elec.g_D)
        
        if abs(FVRT_Ca) > 1e-5:
            J_Roo = (self.elec.J_R * (Ca_SR - Ca_i +
                      J_L / self.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                      (Ca_SR - self.elec.Ca_o * exp_FVRT_Ca)) /
                     (1.0 + self.elec.J_R / self.elec.g_D + J_L / self.elec.g_D *
                      FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        else:
            J_Roo = (self.elec.J_R * (Ca_SR - Ca_i +
                      J_L / self.elec.g_D * 0.00001 / (1.0 - self.exp_00001) *
                      (Ca_SR - self.elec.Ca_o * self.exp_00001)) /
                     (1.0 + self.elec.J_R / self.elec.g_D + J_L / self.elec.g_D *
                      0.00001 / (1.0 - self.exp_00001)))
        
        if abs(FVRT_Ca) > 1e-5:
            J_Loc = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                     (self.elec.Ca_o * exp_FVRT_Ca - Ca_i) /
                     (1.0 + J_L / self.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
            J_Loo = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                     (self.elec.Ca_o * exp_FVRT_Ca - Ca_i +
                      self.elec.J_R / self.elec.g_D * (self.elec.Ca_o * exp_FVRT_Ca - Ca_SR)) /
                     (1.0 + self.elec.J_R / self.elec.g_D + J_L / self.elec.g_D *
                      FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        else:
            J_Loc = (J_L * 0.00001 / (1.0 - self.exp_00001) *
                     (self.elec.Ca_o * self.exp_00001 - Ca_i) /
                     (1.0 + J_L / self.elec.g_D * 0.00001 / (1.0 - self.exp_00001)))
            J_Loo = (J_L * 0.00001 / (1.0 - self.exp_00001) *
                     (self.elec.Ca_o * self.exp_00001 - Ca_i +
                      self.elec.J_R / self.elec.g_D * (self.elec.Ca_o * self.exp_00001 - Ca_SR)) /
                     (1.0 + self.elec.J_R / self.elec.g_D + J_L / self.elec.g_D *
                      0.00001 / (1.0 - self.exp_00001)))
        
        J_L1 = J_Loo * y_oo + J_Loc * y_oc
        J_L2 = J_Loc * alpha_p / (alpha_p + alpha_m)
        
        I_LCC_1 = (z_1 * J_L1 + z_2 * J_L2) * self.elec.N_L_CaRU / V_myo
        
        J_R1 = y_oo * J_Roo + J_Rco * y_co
        J_R3 = J_Rco * beta_pcc / (beta_m + beta_pcc)
        I_RyR_1 = (z_1 * J_R1 + z_3 * J_R3) * self.elec.N_R_CaRU / V_myo
        
        I_LCC_2 = -1.5 * I_LCC_1 * 2.0 * V_myo_uL * F
        I_RyR_2 = 1.5 * I_RyR_1
        
        I_NaCa_2 = (self.elec.g_NCX *
                    (np.exp(self.elec.eta * FVRT) * (Na_i ** 3) * self.elec.Ca_o -
                     np.exp((self.elec.eta - 1.0) * FVRT) * (self.elec.Na_o ** 3) * Ca_i) /
                    (((self.elec.Na_o ** 3) + (self.elec.K_mNa ** 3)) *
                     (self.elec.Ca_o + self.elec.K_mCa) *
                     (1.0 + self.elec.k_sat * np.exp((self.elec.eta - 1.0) * FVRT))))
        I_NaCa_1 = I_NaCa_2 * V_myo_uL * F
        
        I_pCa_2 = self.elec.g_pCa * Ca_i / (self.elec.K_mpCa + Ca_i)
        I_pCa_1 = I_pCa_2 * 2.0 * V_myo_uL * F
        
        E_Ca = R * Tem / (2.0 * F) * np.log(self.elec.Ca_o / (Ca_i + 1e-12))
        I_CaB_2 = self.elec.g_CaB * (E_Ca - V)
        I_CaB_1 = -I_CaB_2 * 2.0 * V_myo_uL * F
        
        I_SERCA = self.elec.g_SERCA * sqr_y8 / (self.elec.K_SERCA ** 2 + sqr_y8)
        I_SR = self.elec.g_SRl * (Ca_SR - Ca_i)
        
        beta_CMDN = 1.0 / (1.0 + self.elec.k_CMDN * self.elec.B_CMDN / ((self.elec.k_CMDN + Ca_i) ** 2))
        
        y_infinity = 1.0 / (1.0 + np.exp((V + 138.6) / 10.48))
        tau_y = 1000.0 / (0.11885 * np.exp((V + 80.0) / 28.37) +
                          0.5623 * np.exp((V + 80.0) / -14.19))
        dy = (y_infinity - y) / tau_y
        
        E_Na = R * Tem / F * np.log(self.elec.Na_o / (Na_i + 1e-12))
        i_Na = self.elec.g_Na_endo * (m_iNa ** 3) * h * j_sod * (V - E_Na)
        i_B_Na = self.elec.g_B_Na * (V - E_Na)
        i_NaK = (self.elec.i_NaK_max *
                 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V * F / (R * Tem)) +
                        0.0365 * self.elec.sigma * np.exp(-V * F / (R * Tem))) *
                 K_o / (K_o + self.elec.K_m_K) *
                 1.0 / (1.0 + ((self.elec.K_m_Na / (Na_i + 1e-12)) ** 4.0)))
        i_f_Na = self.elec.g_f * y * self.elec.f_Na * (V - E_Na)
        
        dNa = -(i_Na + i_B_Na + I_NaCa_1 * 3.0 + i_NaK * 3.0 + i_f_Na) / (V_myo_uL * F)
        
        if (cell_idx == 1 and
            (time % self.elec.stim_period >= self.elec.StimStart_shift) and
            (time % self.elec.stim_period <= self.elec.StimStart_shift + self.elec.stim_duration)):
            I_Stim = self.elec.stim_amplitude
        else:
            I_Stim = 0.0
        
        E_K = R * Tem / F * np.log(K_o / (K_i + 1e-12))
        
        i_ss = self.elec.g_ss * r_ss * s_ss * (V - E_K)
        i_B_K = self.elec.g_B_K * (V - E_K)
        i_t = self.elec.g_t * r1 * (self.elec.a_to * s + self.elec.b_to * s_slow) * (V - E_K)
        
        term1_iK1 = ((48.0e-3 / (np.exp((V + 37.0) / 25.0) +
                                  np.exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 /
                     (1.0 + np.exp((V - (E_K + 76.77)) / -17.0)))
        
        alphaK1 = 10.10001681 / (1.0 + np.exp(-0.69561015 * (V - E_K - 4.148354)))
        betaK1 = ((5.32060272 * np.exp(0.05012022 * (V - E_K + 17.51161017)) +
                   np.exp(0.1 * (V - E_K - 10))) /
                  (1.0 + np.exp(-0.5 * (V - E_K))))
        xK1inf = alphaK1 / (alphaK1 + betaK1 + 1e-12)
        term2_iK1 = self.elec.g_K1 * xK1inf * ((K_o / self.elec.K_o_norm) ** self.elec.deg_Ko_K1) * (V - E_K)
        i_K1 = term1_iK1 + term2_iK1
        
        i_f_K = self.elec.g_f * y * self.f_K * (V - E_K)
        
        P_ATP = 1.0 / (1.0 + (ATP_i / self.elec.V50_P_ATP) ** self.elec.hill_P_ATP)
        i_K_ATP = self.elec.g_K_ATP * P_ATP * (K_o / self.elec.K_o_norm) ** 0.24 * (V - E_K)
        
        dK = -(I_Stim + i_ss + i_B_K + i_t + i_K1 + i_f_K +
               -2.0 * i_NaK + i_K_ATP) / (V_myo_uL * F)
        
        dCa = (beta_CMDN *
               (I_RyR_2 - I_SERCA + I_SR - dTRPN -
                (-2.0 * I_NaCa_1 + I_pCa_1 + I_CaB_1 + I_LCC_2) /
                (2.0 * V_myo_uL * F)))
        dCa_SR = V_myo_uL / V_SR_uL * (-I_RyR_2 + I_SERCA - I_SR)
        
        i_f = i_f_Na + i_f_K
        dV = (-(i_Na + i_t + i_ss + i_f + i_K1 + i_B_Na + i_B_K +
                i_NaK + I_Stim + I_CaB_1 + I_NaCa_1 + I_pCa_1 +
                I_LCC_2 + i_K_ATP) / self.elec.Cm)
        
        h_infinity = 1.0 / (1.0 + np.exp((V + 76.1) / 6.07))
        tau_h = (0.4537 * (1.0 + np.exp(-(V + 10.66) / 11.1)) if V >= -40.0 else
                 3.49 / (0.135 * np.exp(-(V + 80.0) / 6.8) +
                          3.56 * np.exp(0.079 * V) +
                          310000.0 * np.exp(0.35 * V)))
        dh = (h_infinity - h) / tau_h
        
        j_infinity = 1.0 / (1.0 + np.exp((V + 76.1) / 6.07))
        if V >= -40.0:
            tau_j = 11.63 * (1.0 + np.exp(-0.1 * (V + 32.0))) / np.exp(-2.535e-7 * V)
        else:
            tau_j = (3.49 / ((V + 37.78) / (1.0 + np.exp(0.311 * (V + 79.23))) *
                             (-127140.0 * np.exp(0.2444 * V) -
                              3.474e-5 * np.exp(-0.04391 * V)) +
                             0.1212 * np.exp(-0.01052 * V) /
                             (1.0 + np.exp(-0.1378 * (V + 40.14)))))
        dj = (j_infinity - j_sod) / tau_j
        
        m_infinity = 1.0 / (1.0 + np.exp((V + 45.0) / -6.5))
        tau_m = (0.1510178 if V - 47.13 <= 1e-5 else
                 1.36 / (0.32 * (V + 47.13) / (1.0 - np.exp(-0.1 * (V + 47.13))) +
                          0.08 * np.exp(-V / 11.0)))
        dm = (m_infinity - m_iNa) / tau_m
        
        r_infinity = 1.0 / (1.0 + np.exp((V + 10.6) / -11.42))
        tau_r_2 = 100.0 / (45.16 * np.exp(0.03577 * (V + 50.0)) +
                           98.9 * np.exp(-0.1 * (V + 38.0)))
        dr1 = (r_infinity - r1) / tau_r_2
        
        s_infinity = 1.0 / (1.0 + np.exp((V + 45.3) / 6.8841))
        tau_s = 20.0 * np.exp(-((V + 70.0) / 25.0) ** 2) + 35.0
        ds = (s_infinity - s) / tau_s
        
        s_slow_infinity = 1.0 / (1.0 + np.exp((V + 45.3) / 6.8841))
        tau_s_slow = 1300.0 * np.exp(-((V + 70.0) / 30.0) ** 2) + 35.0
        ds_slow = (s_slow_infinity - s_slow) / tau_s_slow
        
        r_ss_infinity = 1.0 / (1.0 + np.exp((V + 11.5) / -11.82))
        tau_r_ss = 10000.0 / (45.16 * np.exp(0.03577 * (V + 50.0)) +
                               98.9 * np.exp(-0.1 * (V + 38.0)))
        dr_ss = (r_ss_infinity - r_ss) / tau_r_ss
        
        s_ss_infinity = 1.0 / (1.0 + np.exp((V + 87.5) / 10.3))
        ds_ss = (s_ss_infinity - s_ss) / self.tau_s_ss
        
        dY = np.zeros(24)
        dY[0], dY[1], dY[2] = dz1, dz2, dz3
        dY[3], dY[4], dY[5] = dr1, ds, ds_slow
        dY[6] = dy
        dY[7], dY[8], dY[9], dY[10], dY[11] = dCa_SR, dCa, dK, dNa, dTRPN
        dY[12] = dV
        dY[17], dY[18], dY[19], dY[20], dY[21] = dh, dj, dm, dr_ss, ds_ss
        
        return dY
    
    def step(self, time: float, Y0: np.ndarray, cell_idx: int, 
              N_elec: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Интегрирует ОДУ на один шаг
        
        Args:
            time: начальное время
            Y0: начальное состояние
            cell_idx: индекс клетки
            N_elec: N для связи
            dt: шаг по времени
        
        Returns:
            Y1: состояние в конце шага
            currents: токи [14]
        """
        def rhs(t, y):
            return self.rhs(t, y, cell_idx, N_elec)
        
        sol = solve_ivp(rhs, [time, time + dt], Y0,
                        method='BDF', rtol=1e-6, atol=1e-8)
        
        if sol.y.size > 0:
            Y1 = sol.y[:, -1]
        else:
            Y1 = Y0.copy()
        
        currents = self._compute_currents(time, Y1, cell_idx, N_elec)
        
        return Y1, currents
    
    def _compute_currents(self, time: float, Y: np.ndarray, 
                          cell_idx: int, N_elec: float) -> np.ndarray:
        """Вычисляет токи для данного состояния"""
        from models.electrical import calculate_ischemia_params
        
        bzdegree = self.ischemia.get_bzdegree(cell_idx)
        
        ATP_i, K_o, g_Na, J_L = calculate_ischemia_params(
            self.ischemia.degree, bzdegree,
            self.elec.ATP_i, self.elec.K_o,
            self.elec.g_Na, self.elec.J_L
        )
        
        V = Y[12]
        Na_i = Y[10]
        K_i = Y[9]
        Ca_i = Y[8]
        Ca_SR = Y[7]
        
        F, R, Tem = self.elec.F, self.elec.R, self.elec.Tem
        V_myo_uL = self.elec.V_myo_uL
        
        FVRT = F * V / (R * Tem)
        
        E_Na = R * Tem / F * np.log(self.elec.Na_o / (Na_i + 1e-12))
        E_K = R * Tem / F * np.log(K_o / (K_i + 1e-12))
        
        i_Na = self.elec.g_Na_endo * (Y[19] ** 3) * Y[17] * Y[18] * (V - E_Na)
        i_B_Na = self.elec.g_B_Na * (V - E_Na)
        i_NaK = (self.elec.i_NaK_max *
                 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V * F / (R * Tem)) +
                        0.0365 * self.elec.sigma * np.exp(-V * F / (R * Tem))) *
                 K_o / (K_o + self.elec.K_m_K) *
                 1.0 / (1.0 + ((self.elec.K_m_Na / (Na_i + 1e-12)) ** 4.0)))
        
        i_f = self.elec.g_f * Y[6] * (self.elec.f_Na * (V - E_Na) + self.f_K * (V - E_K))
        
        I_Stim = (self.elec.stim_amplitude if
                   cell_idx == 1 and
                   time % self.elec.stim_period >= self.elec.StimStart_shift and
                   time % self.elec.stim_period <= self.elec.StimStart_shift + self.elec.stim_duration
                   else 0.0)
        
        i_ss = self.elec.g_ss * Y[20] * Y[21] * (V - E_K)
        i_B_K = self.elec.g_B_K * (V - E_K)
        i_t = self.elec.g_t * Y[3] * (self.elec.a_to * Y[4] + self.elec.b_to * Y[5]) * (V - E_K)
        
        E_Ca = R * Tem / (2.0 * F) * np.log(self.elec.Ca_o / (Ca_i + 1e-12))
        I_CaB_1 = -self.elec.g_CaB * (E_Ca - V) * 2.0 * V_myo_uL * F
        
        I_NaCa_2 = (self.elec.g_NCX *
                    (np.exp(self.elec.eta * FVRT) * (Na_i ** 3) * self.elec.Ca_o -
                     np.exp((self.elec.eta - 1.0) * FVRT) * (self.elec.Na_o ** 3) * Ca_i) /
                    (((self.elec.Na_o ** 3) + (self.elec.K_mNa ** 3)) *
                     (self.elec.Ca_o + self.elec.K_mCa) *
                     (1.0 + self.elec.k_sat * np.exp((self.elec.eta - 1.0) * FVRT))))
        I_NaCa_1 = I_NaCa_2 * V_myo_uL * F
        
        I_pCa_1 = self.elec.g_pCa * Ca_i / (self.elec.K_mpCa + Ca_i) * 2.0 * V_myo_uL * F
        I_LCC_2 = 0.0
        i_K_ATP = 0.0
        
        return np.array([i_Na, i_t, i_ss, i_f, 0, i_B_Na, i_B_K, i_NaK,
                        I_Stim, I_CaB_1, I_NaCa_1, I_pCa_1, I_LCC_2, i_K_ATP])


class DiffusionSolver:
    """Решатель диффузии"""
    
    def __init__(self, params: ModelParameters):
        self.params = params
    
    def step(self, t0: float, ts: float, Yq: np.ndarray, 
             Y0: np.ndarray, elec_solver: ElectricalSolver) -> Tuple[np.ndarray, np.ndarray]:
        """
        Решает уравнение диффузии на один шаг
        
        Args:
            t0: начальное время
            ts: конечное время
            Yq: состояния всех клеток
            Y0: состояние первой клетки
            elec_solver: решатель для первой клетки
        
        Returns:
            Yqn: потенциалы всех клеток
            ZZ: состояние первой клетки
        """
        n = self.params.sim.n
        D = self.params.sim.D_odez
        x0, xn = self.params.sim.x0, self.params.sim.xn
        dx = (xn - x0) / (n - 1)
        dt = (ts - t0)
        
        def rhs(t, y):
            return elec_solver.rhs(t, y, 1, 0.0)
        
        sol = solve_ivp(rhs, [t0, ts], Y0, method='BDF', rtol=1e-6, atol=1e-8)
        ZZ = sol.y[:, -1] if sol.y.size > 0 else Y0
        
        U = np.zeros((n, 2))
        U[:, 0] = Yq[:n, 12]
        
        G = (D * dt / 1000) / (dx) ** 2
        
        a = np.zeros(n)
        b = np.zeros(n)
        
        a[0] = 2 * G / (1 + 2 * G)
        b[0] = ZZ[12] / (1 + 2 * G)
        
        for i in range(1, n):
            denom = -G * a[i-1] + 2 * G + 1
            a[i] = G / denom
            b[i] = (G * b[i-1] + U[i, 0]) / denom
        
        U[n-1, 1] = (b[n-1] + a[n-1] * b[n-2]) / (1 - a[n-1] * a[n-2])
        for i in range(n-2, -1, -1):
            U[i, 1] = a[i] * U[i+1, 1] + b[i]
        
        Yqn = U[:, -1]
        return Yqn, ZZ


class MechanicalSolver:
    """Решатель механической части"""
    
    def __init__(self, params: ModelParameters, ischemia: IschemiaConfig):
        self.params = params
        self.ekb = params.ekb
        self.ischemia = ischemia
    
    def step(self, v_old: float, l1_old: float, l2_old: float, l3_old: float,
             N_old: float, Y: np.ndarray, dt: float, cell_idx: int) -> Tuple[float, float, float]:
        """Один шаг механической модели"""
        from models.mechanical import solve_mechanical
        
        bz = self.ischemia.get_bzdegree(cell_idx)
        
        if self.ischemia.degree == 5:
            Lam_mech = self.ekb.llambda * (1 - 0.43636 * bz)
        elif self.ischemia.degree == 10:
            Lam_mech = self.ekb.llambda * (1 - 0.55 * bz)
        elif self.ischemia.degree == 15:
            Lam_mech = self.ekb.llambda * (1 - 0.8 * bz)
        else:
            Lam_mech = self.ekb.llambda
        
        v_new, l1_new, N_new = solve_mechanical(
            v_old, l1_old, l2_old, l3_old, N_old,
            Y, dt, Lam_mech, self.params
        )
        
        return v_new, l1_new, N_new
