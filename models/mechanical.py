"""
Механическая модель сократительных элементов
"""

import numpy as np
from scipy.optimize import fsolve
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
def compute_q_v(v, v_max, q_1, q_2, q_3, q_4, v_st, beta_Q, alpha_Q):
    """Вычисление q_v - зависимость от скорости"""
    if v <= 0.0:
        return q_1 - q_2 * v / v_max
    elif v <= v_st:
        return (q_4 - q_3) * v / v_st + q_3
    else:
        return q_4 / (1.0 + beta_Q * (v - v_st) / v_max) ** alpha_Q


@jit(nopython=True, cache=True)
def compute_P_star(v, v_max, a, d_h, gamma2):
    """Вычисление P_star"""
    if v <= 0.0:
        return a * (1.0 + v / v_max) / (a - v / v_max)
    else:
        term = v / v_max
        return (
            1.0
            + d_h
            - d_h**2.0 * a / (a * d_h / gamma2 * term**2.0 + (a + 1.0) * term + a * d_h)
        )


@jit(nopython=True, cache=True)
def compute_G_star(v, v_max, a, d_h, alpha_G, alpha_P, v_1, gamma2):
    """Вычисление G_star"""
    term = v / v_max

    if v <= 0.0:
        return 1.0 + 0.6 * term
    elif v <= 0.1:
        P_star = compute_P_star(v, v_max, a, d_h, gamma2)
        return P_star / ((0.4 * a + 1.0) * term / a + 1.0)
    else:
        P_star = compute_P_star(v, v_max, a, d_h, gamma2)
        return (
            P_star
            * np.exp(-alpha_G * ((v - v_1) / v_max) ** alpha_P)
            / ((0.4 * a + 1.0) * term / a + 1.0)
        )


@jit(nopython=True, cache=True)
def compute_p_ext(v, v_max, a, v_1, alpha_G, alpha_P):
    """Вычисление p_ext"""
    term = v / v_max

    if v <= -v_max:
        return 0.0
    elif v <= 0.0:
        return a * (1.0 + term) / ((a - term) * (1.0 + 0.6 * term))
    elif v <= v_1:
        return (0.4 * a + 1.0) * term / a + 1.0
    else:
        return ((0.4 * a + 1.0) * term / a + 1.0) * np.exp(
            alpha_G * (term - 0.1) ** alpha_P
        )


@jit(nopython=True, cache=True)
def compute_n1(l1, g_1, g_2, n1_A, n1_K, n1_C, n1_Q, n1_B, n1_nu):
    """Вычисление n1"""
    denom = (n1_C + n1_Q * np.exp(-n1_B * l1)) ** (1 / n1_nu)
    arg = (g_1 * l1 + g_2) * (n1_A + (n1_K - n1_A) / (denom + 1e-12))

    if arg < 0.0:
        return 0.0
    elif arg < 1.0:
        return arg
    else:
        return 1.0


@jit(nopython=True, cache=True)
def compute_M_A(Ca_ratio, mu, k_mu):
    """Вычисление M_A"""
    return Ca_ratio**mu * (1.0 + k_mu**mu) / (Ca_ratio**mu + k_mu**mu + 1e-12)


@jit(nopython=True, cache=True)
def compute_L_oz(l1, S_0, s055, s046):
    """Вычисление L_oz"""
    if l1 <= s055:
        return (l1 + S_0) / (s046 + S_0 + 1e-12)
    else:
        return (S_0 + s055) / (s046 + S_0 + 1e-12)


def v_meh(v, l1, l2, l3, N_meh, Lam_mech, params):
    """
    Решение для скорости сокращения v

    Аргументы:
        v: скорость сокращения
        l1, l2, l3: длины элементов
        N_meh: N для механической части
        Lam_mech: lambda для механической части
        params: параметры модели

    Возвращает:
        F: невязка
    """
    # Защита от NaN
    if np.isnan(l1) or np.isnan(l2) or np.isnan(l3) or np.isnan(N_meh):
        return 1e10

    N_meh_safe = np.clip(N_meh, 0.0, 1.0)
    l1_safe = max(l1, 0.0001)  # только ограничение снизу
    l2_safe = max(l2, 0.0001)
    l3_safe = max(l3, 0.0001)

    # Вспомогательные переменные
    v_max = params.ekb.v_max
    v_1 = v_max / 10.0

    # Вязкий коэффициент
    if v <= 0.0:
        k_P_vis = params.ekb.beta_vp_l * np.exp(params.ekb.alpha_vp_l * l1_safe)
    else:
        k_P_vis = params.ekb.beta_vp_s * np.exp(params.ekb.alpha_vp_s * l1_safe)

    # p_v
    p_v = compute_p_ext(
        v, v_max, params.ekb.a, v_1, params.ekb.alpha_G, params.ekb.alpha_P
    )

    # Защита от NaN в p_v
    if np.isnan(p_v) or np.isinf(p_v):
        p_v = 1.0

    # Защита от переполнения экспоненты
    exp_arg2 = np.clip(params.ekb.alpha_2 * l2_safe, -700, 700)
    exp_arg3 = np.clip(params.ekb.alpha_3 * l3_safe, -700, 700)

    # Невязка
    F = (
        params.ekb.beta_2 * (np.exp(exp_arg2) - 1.0)
        + Lam_mech * p_v * N_meh_safe
        + k_P_vis * v
        - params.ekb.beta_3 * (np.exp(exp_arg3) - 1.0)
    )

    # Защита от NaN в результате
    if np.isnan(F) or np.isinf(F):
        F = 1e10

    return F


def l2l3(x, l1_n, L, dx, params):
    """
    Решение для длин l2 (для каждой клетки) и l3 (общая)

    Аргументы:
        x: вектор [l2_1, l2_2, ..., l2_n, l3]
        l1_n: массив l1 для всех клеток
        L: общая длина
        dx: шаг по пространству
        params: параметры модели

    Возвращает:
        F: невязка для fsolve
    """
    n = len(l1_n)
    
    alpha_1, beta_1 = params.ekb.alpha_1, params.ekb.beta_1
    alpha_2, beta_2 = params.ekb.alpha_2, params.ekb.beta_2
    alpha_3, beta_3 = params.ekb.alpha_3, params.ekb.beta_3

    x_safe = np.clip(x, -50, 50)
    l1_safe = np.clip(l1_n, -50, 50)
    
    l2_arr = x_safe[:n]
    l3_val = x_safe[n]
    delta = l2_arr - l1_safe
    
    exp_arg2 = np.clip(alpha_2 * l2_arr, -700, 700)
    exp_arg1 = np.clip(alpha_1 * delta, -700, 700)
    exp_arg3 = np.clip(alpha_3 * l3_val, -700, 700)
    
    F = np.empty(n + 1)
    F[:n] = (beta_2 * (np.exp(exp_arg2) - 1.0) +
             beta_1 * (np.exp(exp_arg1) - 1.0) -
             beta_3 * (np.exp(exp_arg3) - 1.0))
    
    F[n] = (np.sum(l2_arr) - (x_safe[0] / 2 + x_safe[n - 1] / 2)) * dx + x_safe[n] - L

    return F


def solve_mechanical(
    v_old, l1_old, l2_old, l3_old, N_old, Y_next, dt, Lam_mech, params
):
    """
    Решение механической части для одной клетки

    Аргументы:
        v_old: скорость на предыдущем шаге
        l1_old, l2_old, l3_old: длины на предыдущем шаге
        N_old: N на предыдущем шаге
        Y_next: электрические переменные на следующем шаге
        dt: шаг по времени
        Lam_mech: lambda для механической части
        params: параметры модели

    Возвращает:
        v_new, l1_new, N_new
    """
    v_max = params.ekb.v_max
    v_st = params.ekb.x_st * v_max
    v_1 = v_max / 10.0
    gamma2 = (
        params.ekb.a
        * params.ekb.d_h
        * (0.1) ** 2.0
        / (3.0 * params.ekb.a * params.ekb.d_h - (params.ekb.a + 1.0) * 0.1)
    )

    # q_v
    q_v = compute_q_v(
        v_old,
        v_max,
        params.ekb.q_1,
        params.ekb.q_2,
        params.ekb.q_3,
        params.ekb.q_4,
        v_st,
        params.ekb.beta_Q,
        params.ekb.alpha_Q,
    )

    # P_star и G_star
    P_star = compute_P_star(v_old, v_max, params.ekb.a, params.ekb.d_h, gamma2)
    G_star = compute_G_star(
        v_old,
        v_max,
        params.ekb.a,
        params.ekb.d_h,
        params.ekb.alpha_G,
        params.ekb.alpha_P,
        v_1,
        gamma2,
    )

    # chi
    if v_old <= 0.0:
        chi = params.ekb.chi_1 + params.ekb.chi_2 * v_old / v_max
    else:
        chi = params.ekb.chi_1

    k_p_v = chi * params.ekb.chi_0 * q_v * params.ekb.m_0 * G_star

    # M_A
    Ca_ratio = Y_next[11] / (params.ekb.A_tot + 1e-12)
    M_A = compute_M_A(Ca_ratio, params.ekb.mu, params.ekb.k_mu)

    # n_1
    n_1 = compute_n1(
        l1_old,
        params.ekb.g_1,
        params.ekb.g_2,
        params.ekb.n1_A,
        params.ekb.n1_K,
        params.ekb.n1_C,
        params.ekb.n1_Q,
        params.ekb.n1_B,
        params.ekb.n1_nu,
    )

    # L_oz
    L_oz = compute_L_oz(l1_old, params.ekb.S_0, params.ekb.s055, params.ekb.s046)

    k_m_v = params.ekb.chi_0 * q_v * (1.0 - chi * params.ekb.m_0 * G_star)
    K_chi = k_p_v * M_A * n_1 * L_oz * (1.0 - N_old) - k_m_v * N_old

    N_new = N_old + dt * K_chi
    N_new = np.clip(N_new, 0.0, 1.0)

    # Решение для скорости v
    def v_func(vv):
        return v_meh(vv[0], l1_old, l2_old, l3_old, N_new, Lam_mech, params)

    try:
        v_sol = fsolve(v_func, v_old, xtol=1e-8)
        v_new = v_sol[0]
    except:
        v_new = v_old

    # Обновление l1
    l1_new = l1_old + dt * v_new
    l1_new = max(l1_new, 0.001)  # не может быть слишком маленькой
    l1_new = min(l1_new, 1.0)    # не может быть слишком большой (физический предел)

    # Защита от нефизических значений
    if l1_new > 0.5:  # если l1 слишком большой, остановить движение
        v_new = 0.0
        l1_new = min(l1_old, 0.5)

    return v_new, l1_new, N_new
