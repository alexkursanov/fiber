"""
Модель диффузии потенциала действия
"""
import numpy as np
from scipy.integrate import solve_ivp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def solve_diffusion(t0, ts, x0, xn, n, D, Yq, Y0,
                    params, global_st, tnnpe_func):
    """
    Решает уравнение диффузии для потенциала действия методом прогонки

    Аргументы:
        t0, ts: начальное и конечное время
        x0, xn: координаты начала и конца
        n: количество клеток
        D: коэффициент диффузии
        Yq: начальные условия для всех клеток
        Y0: начальные условия для первой клетки
        params: параметры модели
        global_st: глобальное состояние
        tnnpe_func: функция TNNPE

    Возвращает:
        Yqn: потенциал в каждой клетке в конечный момент
        V_1: состояние первой клетки в конечный момент
    """
    # Интегрируем первую клетку
    def rhs(t, y):
        return tnnpe_func(t, y, 1, global_st.N_elec, params, global_st)

    try:
        sol = solve_ivp(rhs, [t0, ts], Y0,
                        method='BDF', rtol=1e-6, atol=1e-8)
        ZZ = sol.y[:, -1]
    except Exception as e:
        print(f"Ошибка при интегрировании первой клетки: {e}")
        ZZ = Y0

    # Параметры сетки для диффузии
    s = 2
    x = np.linspace(x0, xn, n + 1)
    dx = x[1] - x[0]
    t = np.linspace(t0, ts, s)
    dt = t[1] - t[0]

    # Решение диффузии методом прогонки
    U = np.zeros((n, s))
    U[:, 0] = Yq[:n, 12]  # потенциал V в каждой клетке

    G = (D * dt / 1000) / (dx) ** 2

    a = np.zeros(n)
    b = np.zeros(n)

    # Прямая прогонка
    a[0] = 2 * G / (1 + 2 * G)
    b[0] = ZZ[12] / (1 + 2 * G)

    for i in range(1, n):
        denom = -G * a[i-1] + 2 * G + 1
        a[i] = G / denom
        b[i] = (G * b[i-1] + U[i, 0]) / denom

    # Обратная прогонка
    U[n-1, 1] = (b[n-1] + a[n-1] * b[n-2]) / (1 - a[n-1] * a[n-2])
    for i in range(n-2, -1, -1):
        U[i, 1] = a[i] * U[i+1, 1] + b[i]

    Yqn = U[:, -1]
    return Yqn, ZZ