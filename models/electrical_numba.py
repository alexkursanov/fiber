"""
Numba-оптимизированная версия tnnpe - параметры передаются явно
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def tnnpe_numba_inner(time, Y, jj, N_elec, bzdegree, IschemiaDeg,
                     F, R, Tem, V_myo_uL, V_SR_uL, V_myo,
                     GeneralTnC, BindingConstTnC, C20, qa,
                     V_L, del_VL, t_L, tau_L, t_R, tau_R_1,
                     K_L, K_RyR, a1, b, c, TNP_d,
                     phi_L, phi_R, theta_R, g_D, J_L, J_R,
                     Ca_o, Na_o, g_NCX, k_sat, K_mNa, K_mCa,
                     g_SERCA, K_SERCA, g_SRl, g_pCa, K_mpCa,
                     g_CaB, Cm, g_B_Na, g_B_K, g_Na, g_Na_endo,
                     g_f, f_Na, g_K1, K_m_K, i_NaK_max, K_m_Na,
                     g_ss, g_t, a_to, b_to, stim_amplitude,
                     stim_duration, stim_period, StimStart_shift,
                     g_K_ATP, ATP_i, V50_P_ATP, hill_P_ATP,
                     K_o_norm, deg_Ko_K1, N_L_CaRU, N_R_CaRU,
                     a_eqmin, s_c):
    """Внутренняя функция tnnpe с numba - все параметры передаются явно"""
    z_1 = Y[0]
    z_2 = Y[1]
    z_3 = Y[2]
    r1 = Y[3]
    s = Y[4]
    s_slow = Y[5]
    y = Y[6]
    Ca_SR = Y[7]
    Ca_i = Y[8]
    K_i = Y[9]
    Na_i = Y[10]
    TRPN = Y[11]
    V = Y[12]
    h = Y[17]
    j_sod = Y[18]
    m_iNa = Y[19]
    r_ss = Y[20]
    s_ss = Y[21]
    
    # Ischemia params
    K_o = 5.4
    
    if IschemiaDeg == 5:
        ATP_i *= (1 - 0.2 * bzdegree)
        K_o = 5.4 + 1.885 * bzdegree
        g_Na *= (1 - 0.125 * bzdegree)
        J_L *= (1 - 0.15 * bzdegree)
    elif IschemiaDeg == 10:
        ATP_i *= (1 - 0.37 * bzdegree)
        K_o = 5.4 + 1.885 * bzdegree
        g_Na *= (1 - 0.25 * bzdegree)
        J_L *= (1 - 0.3 * bzdegree)
    elif IschemiaDeg == 15:
        ATP_i *= (1 - 0.53 * bzdegree)
        K_o = 5.4 + 4.5 * bzdegree
        g_Na *= (1 - 0.375 * bzdegree)
        J_L *= (1 - 0.7 * bzdegree)
    
    # TRPN dynamics
    pi_n1 = GeneralTnC * s_c * N_elec / (TRPN + 1e-12)
    
    if pi_n1 <= 0.0:
        piv = 1.0
    elif pi_n1 <= 1.0:
        piv = a_eqmin ** pi_n1
    else:
        piv = a_eqmin
    
    koff = C20 * np.exp(-qa * TRPN) * piv
    dTRPN = BindingConstTnC * (GeneralTnC - TRPN) * Ca_i - koff * TRPN
    
    # CaRU transitions
    expVL = np.exp((V - V_L) / del_VL)
    alpha_p = expVL / (t_L * (expVL + 1.0))
    
    FVRT = F * V / (R * Tem)
    FVRT_Ca = 2.0 * FVRT
    exp_FVRT_Ca = np.exp(-FVRT_Ca)
    
    if abs(FVRT_Ca) > 1e-9:
        C_oc = ((Ca_i + J_L / g_D * Ca_o *
                 FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) /
                (1.0 + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        C_oc = (Ca_i + J_L / g_D * Ca_o) / (1.0 + J_L / g_D)
    
    sqr_coc = C_oc ** 2
    sqr_y8 = Ca_i ** 2
    sqr_ryr = K_RyR ** 2
    
    beta_poc = sqr_coc / (t_R * (sqr_coc + sqr_ryr))
    beta_pcc = sqr_y8 / (t_R * (sqr_y8 + sqr_ryr))
    
    C_co = (Ca_i + J_R / g_D * Ca_SR) / (1.0 + J_R / g_D)
    
    epsilon_tmp = ((expVL + a1) /
                   (tau_L * K_L * (expVL + 1.0)))
    epsilon_pco = C_co * epsilon_tmp
    epsilon_pcc = Ca_i * epsilon_tmp
    epsilon_m = (b * (expVL + a1) /
                 (tau_L * (b * expVL + a1)))
    
    mu_poc = (sqr_coc + c * sqr_ryr) / (tau_R_1 * (sqr_coc + sqr_ryr))
    mu_pcc = (sqr_y8 + c * sqr_ryr) / (tau_R_1 * (sqr_y8 + sqr_ryr))
    mu_moc = (theta_R * TNP_d *
              (sqr_coc + c * sqr_ryr) /
              (tau_R_1 * (TNP_d * sqr_coc + c * sqr_ryr)))
    mu_mcc = (theta_R * TNP_d *
              (sqr_y8 + c * sqr_ryr) /
              (tau_R_1 * (TNP_d * sqr_y8 + c * sqr_ryr)))
    
    alpha_m = phi_L / t_L
    beta_m = phi_R / t_R
    
    denom = 1 / ((alpha_p + alpha_m) *
                 ((alpha_m + beta_m + beta_poc) *
                  (beta_m + beta_poc) +
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
    
    # Calcium fluxes
    y_oo = alpha_p * (beta_poc * (alpha_p + beta_m + beta_pcc) +
                       beta_pcc * alpha_m) * denom
    
    # J_Rco
    J_Rco = J_R * (Ca_SR - Ca_i) / (1.0 + J_R / g_D)
    
    if abs(FVRT_Ca) > 1e-5:
        J_Roo = (J_R *
                 (Ca_SR - Ca_i +
                  J_L / g_D * FVRT_Ca /
                  (1.0 - exp_FVRT_Ca) * (Ca_SR - Ca_o * exp_FVRT_Ca)) /
                 (1.0 + J_R / g_D +
                  J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        exp_00001 = np.exp(-0.00001)
        J_Roo = (J_R *
                 (Ca_SR - Ca_i +
                  J_L / g_D * 0.00001 /
                  (1.0 - exp_00001) * (Ca_SR - Ca_o * exp_00001)) /
                 (1.0 + J_R / g_D +
                  J_L / g_D * 0.00001 / (1.0 - exp_00001)))
    
    if abs(FVRT_Ca) > 1e-5:
        J_Loc = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (Ca_o * exp_FVRT_Ca - Ca_i) /
                 (1.0 + J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        J_Loo = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (Ca_o * exp_FVRT_Ca - Ca_i +
                  J_R / g_D * (Ca_o * exp_FVRT_Ca - Ca_SR)) /
                 (1.0 + J_R / g_D +
                  J_L / g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        exp_00001 = np.exp(-0.00001)
        J_Loc = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (Ca_o * exp_00001 - Ca_i) /
                 (1.0 + J_L / g_D * 0.00001 / (1.0 - exp_00001)))
        J_Loo = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (Ca_o * exp_00001 - Ca_i +
                  J_R / g_D * (Ca_o * exp_00001 - Ca_SR)) /
                 (1.0 + J_R / g_D +
                  J_L / g_D * 0.00001 / (1.0 - exp_00001)))
    
    J_L1 = J_Loo * y_oo + J_Loc * y_oc
    J_L2 = J_Loc * alpha_p / (alpha_p + alpha_m)
    
    I_LCC_1 = (z_1 * J_L1 + z_2 * J_L2) * N_L_CaRU / V_myo
    
    J_R1 = y_oo * J_Roo + J_Rco * y_co
    J_R3 = J_Rco * beta_pcc / (beta_m + beta_pcc)
    I_RyR_1 = (z_1 * J_R1 + z_3 * J_R3) * N_R_CaRU / V_myo
    
    I_LCC_2 = -1.5 * I_LCC_1 * 2.0 * V_myo_uL * F
    I_RyR_2 = 1.5 * I_RyR_1
    
    # Na-Ca exchanger
    eta = 0.35
    I_NaCa_2 = (g_NCX *
                (np.exp(eta * FVRT) *
                 (Na_i ** 3) * Ca_o -
                 np.exp((eta - 1.0) * FVRT) *
                 (Na_o ** 3) * Ca_i) /
                (((Na_o ** 3) + (K_mNa ** 3)) *
                 (Ca_o + K_mCa) *
                 (1.0 + k_sat *
                  np.exp((eta - 1.0) * FVRT))))
    I_NaCa_1 = I_NaCa_2 * V_myo_uL * F
    
    # Ca pump
    I_pCa_2 = g_pCa * Ca_i / (K_mpCa + Ca_i)
    I_pCa_1 = I_pCa_2 * 2.0 * V_myo_uL * F
    
    # Ca background
    E_Ca = R * Tem / (2.0 * F) * np.log(Ca_o / (Ca_i + 1e-12))
    I_CaB_2 = g_CaB * (E_Ca - V)
    I_CaB_1 = -I_CaB_2 * 2.0 * V_myo_uL * F
    
    # SERCA and SR leak
    I_SERCA = g_SERCA * sqr_y8 / (K_SERCA ** 2 + sqr_y8)
    I_SR = g_SRl * (Ca_SR - Ca_i)
    
    # Buffers
    k_CMDN = 0.002382
    B_CMDN = 0.05
    beta_CMDN = 1.0 / (1.0 + k_CMDN * B_CMDN / ((k_CMDN + Ca_i) ** 2))
    
    # y-gate for funny current
    y_infinity = 1.0 / (1.0 + np.exp((V + 138.6) / 10.48))
    tau_y = 1000.0 / (0.11885 * np.exp((V + 80.0) / 28.37) +
                      0.5623 * np.exp((V + 80.0) / -14.19))
    dy = (y_infinity - y) / tau_y
    
    # Na and K currents
    E_Na = R * Tem / F * np.log(Na_o / (Na_i + 1e-12))
    i_Na = g_Na_endo * (m_iNa ** 3) * h * j_sod * (V - E_Na)
    i_B_Na = g_B_Na * (V - E_Na)
    
    sigma = (np.exp(Na_o / 67.3) - 1.0) / 7.0
    i_NaK = (i_NaK_max *
             1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V * F / (R * Tem)) +
                    0.0365 * sigma * np.exp(-V * F / (R * Tem))) *
             K_o / (K_o + K_m_K) *
             1.0 / (1.0 + ((K_m_Na / (Na_i + 1e-12)) ** 4.0)))
    i_f_Na = g_f * y * f_Na * (V - E_Na)
    
    dNa = -(i_Na + i_B_Na + I_NaCa_1 * 3.0 + i_NaK * 3.0 + i_f_Na) / (V_myo_uL * F)
    
    # Stimulus
    phase = time % stim_period
    if jj == 1 and phase >= StimStart_shift and phase <= StimStart_shift + stim_duration:
        I_Stim = stim_amplitude
    else:
        I_Stim = 0.0
    
    # K currents
    E_K = R * Tem / F * np.log(K_o / (K_i + 1e-12))
    
    i_ss = g_ss * r_ss * s_ss * (V - E_K)
    i_B_K = g_B_K * (V - E_K)
    i_t = g_t * r1 * (a_to * s + b_to * s_slow) * (V - E_K)
    
    # i_K1 (TNNP style)
    term1_iK1 = ((48.0e-3 / (np.exp((V + 37.0) / 25.0) +
                              np.exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 /
                 (1.0 + np.exp((V - (E_K + 76.77)) / -17.0)))
    
    alphaK1 = 10.10001681 / (1 + np.exp(-0.69561015 * (V - E_K - 4.148354)))
    betaK1 = ((5.32060272 * np.exp(0.05012022 * (V - E_K + 17.51161017)) +
               np.exp(0.1 * (V - E_K - 10))) /
              (1 + np.exp(-0.5 * (V - E_K))))
    xK1inf = alphaK1 / (alphaK1 + betaK1 + 1e-12)
    term2_iK1 = (g_K1 * xK1inf *
                 ((K_o / K_o_norm) ** deg_Ko_K1) *
                 (V - E_K))
    i_K1 = term1_iK1 + term2_iK1
    
    f_K = 1.0 - f_Na
    i_f_K = g_f * y * f_K * (V - E_K)
    
    # K_ATP current
    P_ATP = 1.0 / (1.0 + (ATP_i / V50_P_ATP) ** hill_P_ATP)
    i_K_ATP = g_K_ATP * P_ATP * (K_o / K_o_norm) ** 0.24 * (V - E_K)
    
    dK = -(I_Stim + i_ss + i_B_K + i_t + i_K1 + i_f_K +
           -2.0 * i_NaK + i_K_ATP) / (V_myo_uL * F)
    
    # Ca and SR
    dCa = (beta_CMDN *
           (I_RyR_2 - I_SERCA + I_SR - dTRPN -
            (-2.0 * I_NaCa_1 + I_pCa_1 + I_CaB_1 + I_LCC_2) /
            (2.0 * V_myo_uL * F)))
    dCa_SR = V_myo_uL / V_SR_uL * (-I_RyR_2 + I_SERCA - I_SR)
    
    # Voltage
    i_f = i_f_Na + i_f_K
    dV = (-(i_Na + i_t + i_ss + i_f + i_K1 + i_B_Na + i_B_K +
            i_NaK + I_Stim + I_CaB_1 + I_NaCa_1 + I_pCa_1 +
            I_LCC_2 + i_K_ATP) / Cm)
    
    # Na channel gates
    h_infinity = 1.0 / (1.0 + np.exp((V + 76.1) / 6.07))
    if V >= -40.0:
        tau_h = 0.4537 * (1.0 + np.exp(-(V + 10.66) / 11.1))
    else:
        tau_h = (3.49 / (0.135 * np.exp(-(V + 80.0) / 6.8) +
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
    if V - 47.13 <= 1e-5:
        tau_m = 0.1510178
    else:
        tau_m = (1.36 / (0.32 * (V + 47.13) /
                         (1.0 - np.exp(-0.1 * (V + 47.13))) +
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
    tau_s_ss = 2100.0
    ds_ss = (s_ss_infinity - s_ss) / tau_s_ss
    
    # Собираем производные
    dY = np.zeros(24)
    dY[0] = dz1
    dY[1] = dz2
    dY[2] = dz3
    dY[3] = dr1
    dY[4] = ds
    dY[5] = ds_slow
    dY[6] = dy
    dY[7] = dCa_SR
    dY[8] = dCa
    dY[9] = dK
    dY[10] = dNa
    dY[11] = dTRPN
    dY[12] = dV
    dY[17] = dh
    dY[18] = dj
    dY[19] = dm
    dY[20] = dr_ss
    dY[21] = ds_ss
    
    return dY


def tnnpe_numba(time, Y, jj, N_elec, params, global_st):
    """Обертка для numba-функции"""
    bzdegree = global_st.get_bzdegree_for_cell(jj)
    
    e = params.elec
    ekb = params.ekb
    
    return tnnpe_numba_inner(
        time, Y, jj, N_elec, bzdegree, global_st.IschemiaDeg,
        e.F, e.R, e.Tem, e.V_myo_uL, e.V_SR_uL, e.V_myo,
        e.GeneralTnC, e.BindingConstTnC, e.C20, e.qa,
        e.V_L, e.del_VL, e.t_L, e.tau_L, e.t_R, e.tau_R_1,
        e.K_L, e.K_RyR, e.a1, e.b, e.c, e.TNP_d,
        e.phi_L, e.phi_R, e.theta_R, e.g_D, e.J_L, e.J_R,
        e.Ca_o, e.Na_o, e.g_NCX, e.k_sat, e.K_mNa, e.K_mCa,
        e.g_SERCA, e.K_SERCA, e.g_SRl, e.g_pCa, e.K_mpCa,
        e.g_CaB, e.Cm, e.g_B_Na, e.g_B_K, e.g_Na, e.g_Na_endo,
        e.g_f, e.f_Na, e.g_K1, e.K_m_K, e.i_NaK_max, e.K_m_Na,
        e.g_ss, e.g_t, e.a_to, e.b_to, e.stim_amplitude,
        e.stim_duration, e.stim_period, e.StimStart_shift,
        e.g_K_ATP, e.ATP_i, e.V50_P_ATP, e.hill_P_ATP,
        e.K_o_norm, e.deg_Ko_K1, e.N_L_CaRU, e.N_R_CaRU,
        ekb.a_eqmin, ekb.s_c
    )
