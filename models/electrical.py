"""
Электрофизиологическая модель клетки (TNNPE)
Полная реализация на основе MATLAB кода
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Попытка импорта numba для ускорения
try:
    from numba import jit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print("Numba не установлен. Работаем без JIT-ускорения.")
    # Создаем заглушку для декоратора jit
    def jit(nopython=True, cache=True):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def calculate_ischemia_params(IschemiaDeg, bzdegree, ATP_i, K_o, g_Na, J_L):
    """Применяет изменения параметров при ишемии"""
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

    return ATP_i, K_o, g_Na, J_L


def tnnpe(time, Y, jj, N_elec, params, global_st, cache=None):
    """
    Правая часть системы ОДУ электрофизиологической модели

    Аргументы:
        time: текущее время (мс)
        Y: вектор состояния [24 переменные]
        jj: индекс текущей клетки
        N_elec: N для электрической части
        params: параметры модели
        global_st: глобальное состояние
        cache: ParameterCache для оптимизации (опционально)

    Возвращает:
        dYdt: производные
    """
    # Используем IschemiaConfig для вычисления bzdegree
    bzdegree = global_st.get_bzdegree_for_cell(jj)
    
    # Получаем кэш (создаём если не передан)
    if cache is None:
        from core.results import ParameterCache
        cache = ParameterCache.from_params(params)

    # Копируем параметры для возможного изменения
    ATP_i = params.elec.ATP_i
    K_o = params.elec.K_o
    g_Na = params.elec.g_Na
    J_L = params.elec.J_L

    # Применяем изменения при ишемии
    ATP_i, K_o, g_Na, J_L = calculate_ischemia_params(
        global_st.IschemiaDeg, bzdegree, ATP_i, K_o, g_Na, J_L
    )

    # Извлечение переменных состояния
    z_1 = Y[0]   # 'z_1 (dimensionless) (in CaRU_reduced_states)'
    z_2 = Y[1]   # 'z_2 (dimensionless) (in CaRU_reduced_states)'
    z_3 = Y[2]   # 'z_3 (dimensionless) (in CaRU_reduced_states)'
    r1 = Y[3]    # 'r1 (dimensionless) (r in Ca_independent_transient_outward_K_current_r_gate)'
    s = Y[4]     # 's (dimensionless) (in Ca_independent_transient_outward_K_current_s_gate)'
    s_slow = Y[5] # 's_slow (dimensionless) (in Ca_independent_transient_outward_K_current_s_slow_gate)'
    y = Y[6]     # 'y (dimensionless) (in hyperpolarisation_activated_current_y_gate)'
    Ca_SR = Y[7] # 'Ca_SR (mM) (in intracellular_ion_concentrations)'
    Ca_i = Y[8]  # 'Ca_i (mM) (in intracellular_ion_concentrations)'
    K_i = Y[9]   # 'K_i (mM) (in intracellular_ion_concentrations)'
    Na_i = Y[10] # 'Na_i (mM) (in intracellular_ion_concentrations)'
    TRPN = Y[11] # 'TRPN (mM) (in intracellular_ion_concentrations)'
    V = Y[12]    # 'V (mV) (in membrane)'
    h = Y[17]    # 'h (dimensionless) (in sodium_current_h_gate)'
    j_sod = Y[18] # 'j (dimensionless) (in sodium_current_j_gate)'
    m_iNa = Y[19] # 'm (dimensionless) (in sodium_current_m_gate)'
    r_ss = Y[20] # 'r_ss (dimensionless) (in steady_state_outward_K_current_r_ss_gate)'
    s_ss = Y[21] # 's_ss (dimensionless) (in steady_state_outward_K_current_s_ss_gate)'

    # Константы для удобства
    F = params.elec.F
    R = params.elec.R
    Tem = params.elec.Tem
    V_myo_uL = params.elec.V_myo_uL
    V_SR_uL = params.elec.V_SR_uL
    V_myo = params.elec.V_myo

    # Константы из EKB (из кэша)
    a_eqmin = cache.a_eqmin
    s_c = cache.s_c

    # Вспомогательные константы (из кэша)
    exp_00001 = cache.exp_00001
    sqr_ryr = cache.sqr_ryr
    f_K = cache.f_K
    tau_s_ss = cache.tau_s_ss
    
    # Остальные константы
    t_R = params.elec.t_R
    alpha_m = params.elec.alpha_m
    beta_m = params.elec.beta_m
    g_Na_endo = params.elec.g_Na_endo
    sigma = params.elec.sigma

    # TRPN dynamics
    pi_n1 = params.elec.GeneralTnC * s_c * N_elec / (TRPN + 1e-12)

    if pi_n1 <= 0.0:
        piv = 1.0
    elif pi_n1 <= 1.0:
        piv = a_eqmin ** pi_n1
    else:
        piv = a_eqmin

    koff = params.elec.C20 * np.exp(-params.elec.qa * TRPN) * piv
    dTRPN = (params.elec.BindingConstTnC *
             (params.elec.GeneralTnC - TRPN) * Ca_i - koff * TRPN)

    # CaRU transitions
    expVL = np.exp((V - params.elec.V_L) / params.elec.del_VL)
    alpha_p = expVL / (params.elec.t_L * (expVL + 1.0))

    FVRT = F * V / (R * Tem)
    FVRT_Ca = 2.0 * FVRT
    exp_FVRT_Ca = np.exp(-FVRT_Ca)

    # C_oc
    if abs(FVRT_Ca) > 1e-9:
        C_oc = ((Ca_i + J_L / params.elec.g_D * params.elec.Ca_o *
                 FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) /
                (1.0 + J_L / params.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        C_oc = (Ca_i + J_L / params.elec.g_D * params.elec.Ca_o) / \
               (1.0 + J_L / params.elec.g_D)

    sqr_coc = C_oc ** 2
    sqr_y8 = Ca_i ** 2

    beta_poc = sqr_coc / (t_R * (sqr_coc + sqr_ryr))
    beta_pcc = sqr_y8 / (t_R * (sqr_y8 + sqr_ryr))

    C_co = (Ca_i + params.elec.J_R / params.elec.g_D * Ca_SR) / \
           (1.0 + params.elec.J_R / params.elec.g_D)

    epsilon_tmp = ((expVL + params.elec.a1) /
                   (params.elec.tau_L * params.elec.K_L * (expVL + 1.0)))
    epsilon_pco = C_co * epsilon_tmp
    epsilon_pcc = Ca_i * epsilon_tmp
    epsilon_m = (params.elec.b * (expVL + params.elec.a1) /
                 (params.elec.tau_L * (params.elec.b * expVL + params.elec.a1)))

    mu_poc = (sqr_coc + params.elec.c * sqr_ryr) / \
             (params.elec.tau_R_1 * (sqr_coc + sqr_ryr))
    mu_pcc = (sqr_y8 + params.elec.c * sqr_ryr) / \
             (params.elec.tau_R_1 * (sqr_y8 + sqr_ryr))
    mu_moc = (params.elec.theta_R * params.elec.TNP_d *
              (sqr_coc + params.elec.c * sqr_ryr) /
              (params.elec.tau_R_1 *
               (params.elec.TNP_d * sqr_coc + params.elec.c * sqr_ryr)))
    mu_mcc = (params.elec.theta_R * params.elec.TNP_d *
              (sqr_y8 + params.elec.c * sqr_ryr) /
              (params.elec.tau_R_1 *
               (params.elec.TNP_d * sqr_y8 + params.elec.c * sqr_ryr)))

    denom = 1 / ((alpha_p + alpha_m) *
                 ((alpha_m + beta_m + beta_poc) *
                  (beta_m + beta_pcc) +
                  alpha_p * (beta_m + beta_poc)))

    y_oc = (alpha_p * beta_m *
            (alpha_p + alpha_m + beta_m + beta_pcc) * denom)
    y_cc = (alpha_m * beta_m *
            (alpha_m + alpha_p + beta_m + beta_poc) * denom)

    r_1 = y_oc * mu_poc + y_cc * mu_pcc
    r_2 = (alpha_p * mu_moc + alpha_m * mu_mcc) / (alpha_p + alpha_m)
    r_3 = beta_m * mu_pcc / (beta_m + beta_pcc)
    r_4 = mu_mcc

    y_co = (alpha_m *
            (beta_pcc * (alpha_m + beta_m + beta_poc) +
             beta_poc * alpha_p) * denom)
    r_5 = y_co * epsilon_pco + y_cc * epsilon_pcc
    r_6 = epsilon_m
    r_7 = alpha_m * epsilon_pcc / (alpha_p + alpha_m)
    r_8 = epsilon_m

    z_4 = 1.0 - z_1 - z_2 - z_3

    dz1 = -(r_1 + r_5) * z_1 + r_2 * z_2 + r_6 * z_3
    dz2 = r_1 * z_1 - (r_2 + r_7) * z_2 + r_8 * z_4
    dz3 = r_5 * z_1 - (r_6 + r_3) * z_3 + r_4 * z_4

    # Calcium fluxes
    y_oo = (alpha_p *
            (beta_poc * (alpha_p + beta_m + beta_pcc) +
             beta_pcc * alpha_m) * denom)
    y_ci = alpha_m / (alpha_p + alpha_m)
    y_oi = alpha_p / (alpha_p + alpha_m)
    y_ic = beta_m / (beta_pcc + beta_m)
    y_io = beta_pcc / (beta_pcc + beta_m)
    y_ii = 1.0 - y_oc - y_co - y_oo - y_cc - y_ci - y_ic - y_oi - y_io

    C_cc = Ca_i

    # J_Rco, J_Roo
    J_Rco = params.elec.J_R * (Ca_SR - Ca_i) / (1.0 + params.elec.J_R / params.elec.g_D)

    if abs(FVRT_Ca) > 1e-5:
        J_Roo = (params.elec.J_R *
                 (Ca_SR - Ca_i +
                  J_L / params.elec.g_D * FVRT_Ca /
                  (1.0 - exp_FVRT_Ca) * (Ca_SR - params.elec.Ca_o * exp_FVRT_Ca)) /
                 (1.0 + params.elec.J_R / params.elec.g_D +
                  J_L / params.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        J_Roo = (params.elec.J_R *
                 (Ca_SR - Ca_i +
                  J_L / params.elec.g_D * 0.00001 /
                  (1.0 - exp_00001) * (Ca_SR - params.elec.Ca_o * exp_00001)) /
                 (1.0 + params.elec.J_R / params.elec.g_D +
                  J_L / params.elec.g_D * 0.00001 / (1.0 - exp_00001)))

    # J_Loc, J_Loo
    if abs(FVRT_Ca) > 1e-5:
        J_Loc = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (params.elec.Ca_o * exp_FVRT_Ca - Ca_i) /
                 (1.0 + J_L / params.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        J_Loo = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (params.elec.Ca_o * exp_FVRT_Ca - Ca_i +
                  params.elec.J_R / params.elec.g_D *
                  (params.elec.Ca_o * exp_FVRT_Ca - Ca_SR)) /
                 (1.0 + params.elec.J_R / params.elec.g_D +
                  J_L / params.elec.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        J_Loc = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (params.elec.Ca_o * exp_00001 - Ca_i) /
                 (1.0 + J_L / params.elec.g_D * 0.00001 / (1.0 - exp_00001)))
        J_Loo = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (params.elec.Ca_o * exp_00001 - Ca_i +
                  params.elec.J_R / params.elec.g_D *
                  (params.elec.Ca_o * exp_00001 - Ca_SR)) /
                 (1.0 + params.elec.J_R / params.elec.g_D +
                  J_L / params.elec.g_D * 0.00001 / (1.0 - exp_00001)))

    J_L1 = J_Loo * y_oo + J_Loc * y_oc
    J_L2 = J_Loc * alpha_p / (alpha_p + alpha_m)

    I_LCC_1 = (z_1 * J_L1 + z_2 * J_L2) * params.elec.N_L_CaRU / V_myo

    J_R1 = y_oo * J_Roo + J_Rco * y_co
    J_R3 = J_Rco * beta_pcc / (beta_m + beta_pcc)
    I_RyR_1 = (z_1 * J_R1 + z_3 * J_R3) * params.elec.N_R_CaRU / V_myo

    I_LCC_2 = -1.5 * I_LCC_1 * 2.0 * V_myo_uL * F
    I_RyR_2 = 1.5 * I_RyR_1

    # Na-Ca exchanger
    I_NaCa_2 = (params.elec.g_NCX *
                (np.exp(params.elec.eta * FVRT) *
                 (Na_i ** 3) * params.elec.Ca_o -
                 np.exp((params.elec.eta - 1.0) * FVRT) *
                 (params.elec.Na_o ** 3) * Ca_i) /
                (((params.elec.Na_o ** 3) + (params.elec.K_mNa ** 3)) *
                 (params.elec.Ca_o + params.elec.K_mCa) *
                 (1.0 + params.elec.k_sat *
                  np.exp((params.elec.eta - 1.0) * FVRT))))
    I_NaCa_1 = I_NaCa_2 * V_myo_uL * F

    # Ca pump
    I_pCa_2 = params.elec.g_pCa * Ca_i / (params.elec.K_mpCa + Ca_i)
    I_pCa_1 = I_pCa_2 * 2.0 * V_myo_uL * F

    # Ca background
    E_Ca = R * Tem / (2.0 * F) * np.log(params.elec.Ca_o / (Ca_i + 1e-12))
    I_CaB_2 = params.elec.g_CaB * (E_Ca - V)
    I_CaB_1 = -I_CaB_2 * 2.0 * V_myo_uL * F

    # SERCA and SR leak
    I_SERCA = params.elec.g_SERCA * sqr_y8 / \
              (params.elec.K_SERCA ** 2 + sqr_y8)
    I_SR = params.elec.g_SRl * (Ca_SR - Ca_i)

    # Buffers
    beta_CMDN = 1.0 / (1.0 + params.elec.k_CMDN * params.elec.B_CMDN /
                       ((params.elec.k_CMDN + Ca_i) ** 2))

    # y-gate for funny current
    y_infinity = 1.0 / (1.0 + np.exp((V + 138.6) / 10.48))
    tau_y = 1000.0 / (0.11885 * np.exp((V + 80.0) / 28.37) +
                      0.5623 * np.exp((V + 80.0) / -14.19))
    dy = (y_infinity - y) / tau_y

    # Na and K currents
    E_Na = R * Tem / F * np.log(params.elec.Na_o / (Na_i + 1e-12))
    i_Na = g_Na_endo * (m_iNa ** 3) * h * j_sod * (V - E_Na)
    i_B_Na = params.elec.g_B_Na * (V - E_Na)
    i_NaK = (params.elec.i_NaK_max *
             1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V * F / (R * Tem)) +
                    0.0365 * sigma * np.exp(-V * F / (R * Tem))) *
             K_o / (K_o + params.elec.K_m_K) *
             1.0 / (1.0 + ((params.elec.K_m_Na / (Na_i + 1e-12)) ** 4.0)))
    i_f_Na = params.elec.g_f * y * params.elec.f_Na * (V - E_Na)

    dNa = -(i_Na + i_B_Na + I_NaCa_1 * 3.0 + i_NaK * 3.0 + i_f_Na) / \
          (V_myo_uL * F)

    # Stimulus
    if (jj == 1 and
        (time % params.elec.stim_period >= params.elec.StimStart_shift) and
        (time % params.elec.stim_period <=
         params.elec.StimStart_shift + params.elec.stim_duration)):
        I_Stim = params.elec.stim_amplitude
    else:
        I_Stim = 0.0

    # K currents
    E_K = R * Tem / F * np.log(K_o / (K_i + 1e-12))

    i_ss = params.elec.g_ss * r_ss * s_ss * (V - E_K)
    i_B_K = params.elec.g_B_K * (V - E_K)
    i_t = (params.elec.g_t * r1 *
           (params.elec.a_to * s + params.elec.b_to * s_slow) * (V - E_K))

    # i_K1 (TNNP style)
    term1_iK1 = ((48.0e-3 / (np.exp((V + 37.0) / 25.0) +
                              np.exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 /
                 (1.0 + np.exp((V - (E_K + 76.77)) / -17.0)))

    alphaK1 = 10.10001681 / (1 + np.exp(-0.69561015 * (V - E_K - 4.148354)))
    betaK1 = ((5.32060272 * np.exp(0.05012022 * (V - E_K + 17.51161017)) +
               np.exp(0.1 * (V - E_K - 10))) /
              (1 + np.exp(-0.5 * (V - E_K))))
    xK1inf = alphaK1 / (alphaK1 + betaK1 + 1e-12)
    term2_iK1 = (params.elec.g_K1 * xK1inf *
                 ((K_o / params.elec.K_o_norm) ** params.elec.deg_Ko_K1) *
                 (V - E_K))
    i_K1 = term1_iK1 + term2_iK1

    i_f_K = params.elec.g_f * y * f_K * (V - E_K)

    # K_ATP current
    P_ATP = 1.0 / (1.0 + (ATP_i / params.elec.V50_P_ATP) **
                   params.elec.hill_P_ATP)
    i_K_ATP = (params.elec.g_K_ATP * P_ATP *
               (K_o / params.elec.K_o_norm) ** 0.24 * (V - E_K))

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
            I_LCC_2 + i_K_ATP) / params.elec.Cm)

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
        tau_j = 11.63 * (1.0 + np.exp(-0.1 * (V + 32.0))) / \
                np.exp(-2.535e-7 * V)
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
    ds_ss = (s_ss_infinity - s_ss) / tau_s_ss

    # Собираем производные
    dY = np.zeros(24)
    dY[0] = dz1      # z_1
    dY[1] = dz2      # z_2
    dY[2] = dz3      # z_3
    dY[3] = dr1      # r1
    dY[4] = ds       # s
    dY[5] = ds_slow  # s_slow
    dY[6] = dy       # y
    dY[7] = dCa_SR   # Ca_SR
    dY[8] = dCa      # Ca_i
    dY[9] = dK       # K_i
    dY[10] = dNa     # Na_i
    dY[11] = dTRPN   # TRPN
    dY[12] = dV      # V
    dY[13] = 0.0     # Q_1 (не используется)
    dY[14] = 0.0     # Q_2 (не используется)
    dY[15] = 0.0     # Q_3 (не используется)
    dY[16] = 0.0     # z (не используется)
    dY[17] = dh      # h
    dY[18] = dj      # j_sod
    dY[19] = dm      # m_iNa
    dY[20] = dr_ss   # r_ss
    dY[21] = ds_ss   # s_ss
    dY[22] = 0.0     # B1 (не используется)
    dY[23] = 0.0     # B2 (не используется)

    # Сохраняем токи в глобальном состоянии
    cell_cur = np.array([
        i_Na, i_t, i_ss, i_f, i_K1, i_B_Na, i_B_K, i_NaK,
        I_Stim, I_CaB_1, I_NaCa_1, I_pCa_1, I_LCC_2, i_K_ATP
    ])

    global_st.cell_cur = cell_cur

    return dY


def tnnpe_explicit(time, Y, cell_idx, N_elec, elec_params, ekb_params, ischemia):
    """
    Правая часть системы ОДУ электрофизиологической модели (явная версия)
    
    Эта версия принимает параметры явно, без объекта global_st.
    Использует dataclass IschemiaConfig для конфигурации ишемии.
    
    Аргументы:
        time: текущее время (мс)
        Y: вектор состояния [24 переменные]
        cell_idx: индекс текущей клетки (1-based)
        N_elec: N для электрической части
        elec_params: ElectricalParameters
        ekb_params: EKBParameters  
        ischemia: IschemiaConfig или просто ischemia_degree (int)
    
    Возвращает:
        dYdt: производные
        currents: массив токов [14]
    """
    # Обработка ischemia - может быть int или IschemiaConfig
    if isinstance(ischemia, int):
        ischemia_degree = ischemia
        bzdegree = 0.0  # Без пространственной зависимости
    else:
        ischemia_degree = ischemia.degree
        bzdegree = ischemia.get_bzdegree(cell_idx)
    
    # Применяем изменения при ишемии
    ATP_i = elec_params.ATP_i
    K_o = elec_params.K_o
    g_Na = elec_params.g_Na
    J_L = elec_params.J_L
    
    ATP_i, K_o, g_Na, J_L = calculate_ischemia_params(
        ischemia_degree, bzdegree, ATP_i, K_o, g_Na, J_L
    )
    
    # Извлечение переменных состояния
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
    
    # Константы
    F = elec_params.F
    R = elec_params.R
    Tem = elec_params.Tem
    V_myo_uL = elec_params.V_myo_uL
    V_SR_uL = elec_params.V_SR_uL
    V_myo = elec_params.V_myo
    
    # Константы из EKB
    a_eqmin = ekb_params.a_eqmin
    s_c = ekb_params.s_c
    
    # Вспомогательные константы
    exp_00001 = np.exp(-0.00001)
    t_R = elec_params.t_R
    alpha_m = elec_params.alpha_m
    beta_m = elec_params.beta_m
    g_Na_endo = elec_params.g_Na_endo
    sigma = elec_params.sigma
    f_K = elec_params.f_K
    tau_s_ss = elec_params.tau_s_ss
    sqr_ryr = elec_params.K_RyR ** 2
    
    # TRPN dynamics
    pi_n1 = elec_params.GeneralTnC * s_c * N_elec / (TRPN + 1e-12)
    
    if pi_n1 <= 0.0:
        piv = 1.0
    elif pi_n1 <= 1.0:
        piv = a_eqmin ** pi_n1
    else:
        piv = a_eqmin
    
    koff = elec_params.C20 * np.exp(-elec_params.qa * TRPN) * piv
    dTRPN = (elec_params.BindingConstTnC *
             (elec_params.GeneralTnC - TRPN) * Ca_i - koff * TRPN)
    
    # CaRU transitions
    expVL = np.exp((V - elec_params.V_L) / elec_params.del_VL)
    alpha_p = expVL / (elec_params.t_L * (expVL + 1.0))
    
    FVRT = F * V / (R * Tem)
    FVRT_Ca = 2.0 * FVRT
    exp_FVRT_Ca = np.exp(-FVRT_Ca)
    
    # C_oc
    if abs(FVRT_Ca) > 1e-9:
        C_oc = ((Ca_i + J_L / elec_params.g_D * elec_params.Ca_o *
                 FVRT_Ca * exp_FVRT_Ca / (1.0 - exp_FVRT_Ca)) /
                (1.0 + J_L / elec_params.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        C_oc = (Ca_i + J_L / elec_params.g_D * elec_params.Ca_o) / \
               (1.0 + J_L / elec_params.g_D)
    
    sqr_coc = C_oc ** 2
    sqr_y8 = Ca_i ** 2
    
    beta_poc = sqr_coc / (t_R * (sqr_coc + sqr_ryr))
    beta_pcc = sqr_y8 / (t_R * (sqr_y8 + sqr_ryr))
    
    C_co = (Ca_i + elec_params.J_R / elec_params.g_D * Ca_SR) / \
           (1.0 + elec_params.J_R / elec_params.g_D)
    
    epsilon_tmp = ((expVL + elec_params.a1) /
                   (elec_params.tau_L * elec_params.K_L * (expVL + 1.0)))
    epsilon_pco = C_co * epsilon_tmp
    epsilon_pcc = Ca_i * epsilon_tmp
    epsilon_m = (elec_params.b * (expVL + elec_params.a1) /
                 (elec_params.tau_L * (elec_params.b * expVL + elec_params.a1)))
    
    mu_poc = (sqr_coc + elec_params.c * sqr_ryr) / \
             (elec_params.tau_R_1 * (sqr_coc + sqr_ryr))
    mu_pcc = (sqr_y8 + elec_params.c * sqr_ryr) / \
             (elec_params.tau_R_1 * (sqr_y8 + sqr_ryr))
    mu_moc = (elec_params.theta_R * elec_params.TNP_d *
              (sqr_coc + elec_params.c * sqr_ryr) /
              (elec_params.tau_R_1 *
               (elec_params.TNP_d * sqr_coc + elec_params.c * sqr_ryr)))
    mu_mcc = (elec_params.theta_R * elec_params.TNP_d *
              (sqr_y8 + elec_params.c * sqr_ryr) /
              (elec_params.tau_R_1 *
               (elec_params.TNP_d * sqr_y8 + elec_params.c * sqr_ryr)))
    
    denom = 1 / ((alpha_p + alpha_m) *
                 ((alpha_m + beta_m + beta_poc) *
                  (beta_m + beta_pcc) +
                  alpha_p * (beta_m + beta_poc)))
    
    y_oc = (alpha_p * beta_m *
            (alpha_p + alpha_m + beta_m + beta_pcc) * denom)
    y_cc = (alpha_m * beta_m *
            (alpha_m + alpha_p + beta_m + beta_poc) * denom)
    
    r_1 = y_oc * mu_poc + y_cc * mu_pcc
    r_2 = (alpha_p * mu_moc + alpha_m * mu_mcc) / (alpha_p + alpha_m)
    r_3 = beta_m * mu_pcc / (beta_m + beta_pcc)
    r_4 = mu_mcc
    
    y_co = (alpha_m *
            (beta_pcc * (alpha_m + beta_m + beta_poc) +
             beta_poc * alpha_p) * denom)
    r_5 = y_co * epsilon_pco + y_cc * epsilon_pcc
    r_6 = epsilon_m
    r_7 = alpha_m * epsilon_pcc / (alpha_p + alpha_m)
    r_8 = epsilon_m
    
    z_4 = 1.0 - z_1 - z_2 - z_3
    
    dz1 = -(r_1 + r_5) * z_1 + r_2 * z_2 + r_6 * z_3
    dz2 = r_1 * z_1 - (r_2 + r_7) * z_2 + r_8 * z_4
    dz3 = r_5 * z_1 - (r_6 + r_3) * z_3 + r_4 * z_4
    
    # Calcium fluxes
    y_oo = (alpha_p *
            (beta_poc * (alpha_p + beta_m + beta_pcc) +
             beta_pcc * alpha_m) * denom)
    y_ci = alpha_m / (alpha_p + alpha_m)
    y_oi = alpha_p / (alpha_p + alpha_m)
    y_ic = beta_m / (beta_pcc + beta_m)
    y_io = beta_pcc / (beta_pcc + beta_m)
    y_ii = 1.0 - y_oc - y_co - y_oo - y_cc - y_ci - y_ic - y_oi - y_io
    
    C_cc = Ca_i
    
    # J_Rco, J_Roo
    J_Rco = elec_params.J_R * (Ca_SR - Ca_i) / (1.0 + elec_params.J_R / elec_params.g_D)
    
    if abs(FVRT_Ca) > 1e-5:
        J_Roo = (elec_params.J_R *
                 (Ca_SR - Ca_i +
                  J_L / elec_params.g_D * FVRT_Ca /
                  (1.0 - exp_FVRT_Ca) * (Ca_SR - elec_params.Ca_o * exp_FVRT_Ca)) /
                 (1.0 + elec_params.J_R / elec_params.g_D +
                  J_L / elec_params.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        J_Roo = (elec_params.J_R *
                 (Ca_SR - Ca_i +
                  J_L / elec_params.g_D * 0.00001 /
                  (1.0 - exp_00001) * (Ca_SR - elec_params.Ca_o * exp_00001)) /
                 (1.0 + elec_params.J_R / elec_params.g_D +
                  J_L / elec_params.g_D * 0.00001 / (1.0 - exp_00001)))
    
    # J_Loc, J_Loo
    if abs(FVRT_Ca) > 1e-5:
        J_Loc = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (elec_params.Ca_o * exp_FVRT_Ca - Ca_i) /
                 (1.0 + J_L / elec_params.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
        J_Loo = (J_L * FVRT_Ca / (1.0 - exp_FVRT_Ca) *
                 (elec_params.Ca_o * exp_FVRT_Ca - Ca_i +
                  elec_params.J_R / elec_params.g_D *
                  (elec_params.Ca_o * exp_FVRT_Ca - Ca_SR)) /
                 (1.0 + elec_params.J_R / elec_params.g_D +
                  J_L / elec_params.g_D * FVRT_Ca / (1.0 - exp_FVRT_Ca)))
    else:
        J_Loc = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (elec_params.Ca_o * exp_00001 - Ca_i) /
                 (1.0 + J_L / elec_params.g_D * 0.00001 / (1.0 - exp_00001)))
        J_Loo = (J_L * 0.00001 / (1.0 - exp_00001) *
                 (elec_params.Ca_o * exp_00001 - Ca_i +
                  elec_params.J_R / elec_params.g_D *
                  (elec_params.Ca_o * exp_00001 - Ca_SR)) /
                 (1.0 + elec_params.J_R / elec_params.g_D +
                  J_L / elec_params.g_D * 0.00001 / (1.0 - exp_00001)))
    
    J_L1 = J_Loo * y_oo + J_Loc * y_oc
    J_L2 = J_Loc * alpha_p / (alpha_p + alpha_m)
    
    I_LCC_1 = (z_1 * J_L1 + z_2 * J_L2) * elec_params.N_L_CaRU / V_myo
    
    J_R1 = y_oo * J_Roo + J_Rco * y_co
    J_R3 = J_Rco * beta_pcc / (beta_m + beta_pcc)
    I_RyR_1 = (z_1 * J_R1 + z_3 * J_R3) * elec_params.N_R_CaRU / V_myo
    
    I_LCC_2 = -1.5 * I_LCC_1 * 2.0 * V_myo_uL * F
    I_RyR_2 = 1.5 * I_RyR_1
    
    # Na-Ca exchanger
    I_NaCa_2 = (elec_params.g_NCX *
                (np.exp(elec_params.eta * FVRT) *
                 (Na_i ** 3) * elec_params.Ca_o -
                 np.exp((elec_params.eta - 1.0) * FVRT) *
                 (elec_params.Na_o ** 3) * Ca_i) /
                (((elec_params.Na_o ** 3) + (elec_params.K_mNa ** 3)) *
                 (elec_params.Ca_o + elec_params.K_mCa) *
                 (1.0 + elec_params.k_sat *
                  np.exp((elec_params.eta - 1.0) * FVRT))))
    I_NaCa_1 = I_NaCa_2 * V_myo_uL * F
    
    # Ca pump
    I_pCa_2 = elec_params.g_pCa * Ca_i / (elec_params.K_mpCa + Ca_i)
    I_pCa_1 = I_pCa_2 * 2.0 * V_myo_uL * F
    
    # Ca background
    E_Ca = R * Tem / (2.0 * F) * np.log(elec_params.Ca_o / (Ca_i + 1e-12))
    I_CaB_2 = elec_params.g_CaB * (E_Ca - V)
    I_CaB_1 = -I_CaB_2 * 2.0 * V_myo_uL * F
    
    # SERCA and SR leak
    I_SERCA = elec_params.g_SERCA * sqr_y8 / \
              (elec_params.K_SERCA ** 2 + sqr_y8)
    I_SR = elec_params.g_SRl * (Ca_SR - Ca_i)
    
    # Buffers
    beta_CMDN = 1.0 / (1.0 + elec_params.k_CMDN * elec_params.B_CMDN /
                       ((elec_params.k_CMDN + Ca_i) ** 2))
    
    # y-gate for funny current
    y_infinity = 1.0 / (1.0 + np.exp((V + 138.6) / 10.48))
    tau_y = 1000.0 / (0.11885 * np.exp((V + 80.0) / 28.37) +
                      0.5623 * np.exp((V + 80.0) / -14.19))
    dy = (y_infinity - y) / tau_y
    
    # Na and K currents
    E_Na = R * Tem / F * np.log(elec_params.Na_o / (Na_i + 1e-12))
    i_Na = g_Na_endo * (m_iNa ** 3) * h * j_sod * (V - E_Na)
    i_B_Na = elec_params.g_B_Na * (V - E_Na)
    i_NaK = (elec_params.i_NaK_max *
             1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V * F / (R * Tem)) +
                    0.0365 * sigma * np.exp(-V * F / (R * Tem))) *
             K_o / (K_o + elec_params.K_m_K) *
             1.0 / (1.0 + ((elec_params.K_m_Na / (Na_i + 1e-12)) ** 4.0)))
    i_f_Na = elec_params.g_f * y * elec_params.f_Na * (V - E_Na)
    
    dNa = -(i_Na + i_B_Na + I_NaCa_1 * 3.0 + i_NaK * 3.0 + i_f_Na) / \
          (V_myo_uL * F)
    
    # Stimulus
    if (cell_idx == 1 and
        (time % elec_params.stim_period >= elec_params.StimStart_shift) and
        (time % elec_params.stim_period <=
         elec_params.StimStart_shift + elec_params.stim_duration)):
        I_Stim = elec_params.stim_amplitude
    else:
        I_Stim = 0.0
    
    # K currents
    E_K = R * Tem / F * np.log(K_o / (K_i + 1e-12))
    
    i_ss = elec_params.g_ss * r_ss * s_ss * (V - E_K)
    i_B_K = elec_params.g_B_K * (V - E_K)
    i_t = (elec_params.g_t * r1 *
           (elec_params.a_to * s + elec_params.b_to * s_slow) * (V - E_K))
    
    # i_K1 (TNNP style)
    term1_iK1 = ((48.0e-3 / (np.exp((V + 37.0) / 25.0) +
                              np.exp((V + 37.0) / -25.0)) + 10.0e-3) * 0.001 /
                 (1.0 + np.exp((V - (E_K + 76.77)) / -17.0)))
    
    alphaK1 = 10.10001681 / (1 + np.exp(-0.69561015 * (V - E_K - 4.148354)))
    betaK1 = ((5.32060272 * np.exp(0.05012022 * (V - E_K + 17.51161017)) +
               np.exp(0.1 * (V - E_K - 10))) /
              (1 + np.exp(-0.5 * (V - E_K))))
    xK1inf = alphaK1 / (alphaK1 + betaK1 + 1e-12)
    term2_iK1 = (elec_params.g_K1 * xK1inf *
                 ((K_o / elec_params.K_o_norm) ** elec_params.deg_Ko_K1) *
                 (V - E_K))
    i_K1 = term1_iK1 + term2_iK1
    
    i_f_K = elec_params.g_f * y * f_K * (V - E_K)
    
    # K_ATP current
    P_ATP = 1.0 / (1.0 + (ATP_i / elec_params.V50_P_ATP) **
                   elec_params.hill_P_ATP)
    i_K_ATP = (elec_params.g_K_ATP * P_ATP *
               (K_o / elec_params.K_o_norm) ** 0.24 * (V - E_K))
    
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
            I_LCC_2 + i_K_ATP) / elec_params.Cm)
    
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
        tau_j = 11.63 * (1.0 + np.exp(-0.1 * (V + 32.0))) / \
                np.exp(-2.535e-7 * V)
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
    ds_ss = (s_ss_infinity - s_ss) / tau_s_ss
    
    # Собираем производные
    dY = np.zeros(24)
    dY[0] = dz1      # z_1
    dY[1] = dz2      # z_2
    dY[2] = dz3      # z_3
    dY[3] = dr1      # r1
    dY[4] = ds       # s
    dY[5] = ds_slow  # s_slow
    dY[6] = dy       # y
    dY[7] = dCa_SR   # Ca_SR
    dY[8] = dCa      # Ca_i
    dY[9] = dK       # K_i
    dY[10] = dNa     # Na_i
    dY[11] = dTRPN   # TRPN
    dY[12] = dV      # V
    dY[17] = dh      # h
    dY[18] = dj      # j_sod
    dY[19] = dm      # m_iNa
    dY[20] = dr_ss   # r_ss
    dY[21] = ds_ss   # s_ss
    
    # Токи
    currents = np.array([
        i_Na, i_t, i_ss, i_f, i_K1, i_B_Na, i_B_K, i_NaK,
        I_Stim, I_CaB_1, I_NaCa_1, I_pCa_1, I_LCC_2, i_K_ATP
    ])
    
    return dY, currents


def y_init():
    """Начальные условия для всех 24 переменных"""
    Y = np.zeros(24)

    Y[0] = 0.987846977687179   # z_1
    Y[1] = 0.00889630179839756  # z_2
    Y[2] = 0.00322801867751232  # z_3
    Y[3] = 0.002510729584521    # r1
    Y[4] = 0.992513467576987    # s
    Y[5] = 0.695813093535168    # s_slow
    Y[6] = 0.00284312435079205  # y
    Y[7] = 0.742474679225752    # Ca_SR
    Y[8] = 0.000116263904902851 # Ca_i
    Y[9] = 137.631285943762     # K_i
    Y[10] = 12.2406809965944    # Na_i
    Y[11] = 0.00104123225158751 # TRPN
    Y[12] = -78.9449088406309   # V
    Y[13] = 0.0                 # Q_1
    Y[14] = 0.0                 # Q_2
    Y[15] = 0.0                 # Q_3
    Y[16] = 0.00982273628901418 # z (niederer)
    Y[17] = 0.615054357972781   # h
    Y[18] = 0.614459274494406   # j_sod
    Y[19] = 0.00536599273054972 # m_iNa
    Y[20] = 0.00331493233624982 # r_ss
    Y[21] = 0.271740759186668   # s_ss
    Y[22] = 0.0                 # B1
    Y[23] = 0.0                 # B2

    return Y