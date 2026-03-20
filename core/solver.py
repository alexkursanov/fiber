"""
Основной решатель модели
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from tqdm import tqdm
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mechanical import solve_mechanical, l2l3


class CardiacSolver:
    """Основной класс-решатель модели"""

    def __init__(self, params, global_state):
        """
        Инициализация решателя

        Аргументы:
            params: параметры модели
            global_state: глобальное состояние
        """
        self.params = params
        self.gs = global_state

        # Массивы для результатов
        self._init_arrays()

        # Вспомогательные переменные
        self.v_1 = params.ekb.v_max / 10.0
        self.v_st = params.ekb.x_st * params.ekb.v_max
        self.gamma2 = (params.ekb.a * params.ekb.d_h * (0.1) ** 2.0 /
                       (3.0 * params.ekb.a * params.ekb.d_h -
                        (params.ekb.a + 1.0) * 0.1))

        # Инициализация механических переменных
        self._init_mechanical()

    def _init_arrays(self):
        """Инициализация массивов для хранения результатов"""
        s = self.params.sim.s
        n = self.params.sim.n

        self.Y = np.zeros((s, n, 24))
        self.Y1 = np.zeros((n, 24))
        self.cell_currents = np.zeros((s, n, 14))

        self.l_1 = np.zeros((s, n))
        self.l_2 = np.zeros((s, n))
        self.l_3 = np.zeros(s)
        self.N = np.zeros((s, n))
        self.v = np.zeros((s, n))
        self.w = np.zeros((s, n))
        self.deltaU = np.zeros((s, n))

    def _init_mechanical(self):
        """Инициализация механических переменных"""
        r0 = self.params.ekb.r0
        beta_2 = self.params.ekb.beta_2
        alpha_2 = self.params.ekb.alpha_2
        beta_1 = self.params.ekb.beta_1
        alpha_3 = self.params.ekb.alpha_3
        beta_3 = self.params.ekb.beta_3

        self.v[0, :] = -0.00000163008453003026
        self.w[0, :] = 7.62412076632331e-07
        self.l_2[0, :] = np.log((r0 + beta_2) / beta_2) / alpha_2
        self.l_1[0, :] = (self.l_2[0, 0] +
                          (np.log(beta_1) - np.log(r0 + beta_1 -
                           beta_2 * (np.exp(alpha_2 * self.l_2[0, 0]) - 1))) /
                          alpha_2)
        self.l_3[0] = np.log((beta_3 + r0) / beta_3) / alpha_3
        self.N[0, :] = 0.0000284517486098194

        self.gs.L = (self.params.sim.n - 1) * self.l_2[0, 0] * \
                    self.params.sim.dx + self.l_3[0]
        self.gs.l1_n = self.l_1[0, :].copy()

    def set_initial_conditions(self, Y_init):
        """Установка начальных условий"""
        for j in range(self.params.sim.n):
            self.Y[0, j, :] = Y_init

    def solve_step(self, i, tnnpe_func, diffusion_func):
        """
        Решение одного шага по времени

        Аргументы:
            i: индекс текущего шага
            tnnpe_func: функция TNNPE
            diffusion_func: функция диффузии
        """
        dt = self.params.sim.dt
        t_current = self.params.sim.t[i]
        t_next = self.params.sim.t[i+1]

        # -----------------------------------------------------------------
        # 1-я половина электрического цикла
        # -----------------------------------------------------------------
        for jj in range(1, self.params.sim.n):
            self.gs.jj = jj + 1
            self.gs.N_elec = self.N[i, jj]

            def rhs(t, y):
                return tnnpe_func(t, y, self.gs.jj, self.gs.N_elec,
                                  self.params, self.gs)

            try:
                sol = solve_ivp(
                    rhs,
                    [t_current, t_current + dt/2],
                    self.Y[i, jj, :],
                    method='BDF', rtol=1e-6, atol=1e-8
                )
                if sol.y.size > 0:
                    self.Y1[jj, :] = sol.y[:, -1]
                else:
                    self.Y1[jj, :] = self.Y[i, jj, :]
            except Exception as e:
                print(f"Ошибка при интегрировании клетки {jj}: {e}")
                self.Y1[jj, :] = self.Y[i, jj, :]

        # Первая клетка
        self.gs.jj = 1
        self.gs.N_elec = self.N[i, 0]

        # Решение PDE
        try:
            Yqn, V_1 = diffusion_func(
                t_current, t_next,
                self.params.sim.x0, self.params.sim.xn,
                self.params.sim.n, self.params.sim.D_odez,
                self.Y1, self.Y[i, 0, :],
                self.params, self.gs, tnnpe_func
            )

            self.cell_currents[i, 0, :] = self.gs.cell_cur
            self.Y1[:, 12] = Yqn

            # Разность потенциалов между соседними клетками
            for jj in range(1, self.params.sim.n):
                self.deltaU[i, jj] = Yqn[jj] - Yqn[jj-1]
        except Exception as e:
            print(f"Ошибка в diffusion_func: {e}")
            self.Y1[:, 12] = self.Y[i, :, 12]
            V_1 = self.Y[i, 0, :]

        # -----------------------------------------------------------------
        # 2-я половина электрического цикла
        # -----------------------------------------------------------------
        for jj in range(1, self.params.sim.n):
            self.gs.jj = jj + 1
            self.gs.N_elec = self.N[i, jj]

            def rhs(t, y):
                return tnnpe_func(t, y, self.gs.jj, self.gs.N_elec,
                                  self.params, self.gs)

            try:
                sol = solve_ivp(
                    rhs,
                    [t_current + dt/2, t_next],
                    self.Y1[jj, :],
                    method='BDF', rtol=1e-6, atol=1e-8
                )
                if sol.y.size > 0:
                    self.Y[i+1, jj, :] = sol.y[:, -1]
                else:
                    self.Y[i+1, jj, :] = self.Y1[jj, :]
            except Exception as e:
                print(f"Ошибка при интегрировании клетки {jj}: {e}")
                self.Y[i+1, jj, :] = self.Y1[jj, :]

        # Токи сохраняются в gs.cell_cur после каждого вызова tnnpe_func
        # Для первой клетки сохраняем после диффузии
        self.cell_currents[i, 0, :] = self.gs.cell_cur

        self.Y[i+1, 0, :] = V_1

        # -----------------------------------------------------------------
        # Механическая часть
        # -----------------------------------------------------------------
        self._solve_mechanical_step(i)

    def _solve_mechanical_step(self, i):
        """Решение механической части для шага i"""
        dt = self.params.sim.dt

        for j in range(self.params.sim.n):
            # Параметры ишемии для механической части
            bz = self.gs.get_bzdegree_for_cell(j + 1)

            if self.gs.IschemiaDeg == 5:
                Lam_mech = self.params.ekb.llambda * (1 - 0.43636 * bz)
            elif self.gs.IschemiaDeg == 10:
                Lam_mech = self.params.ekb.llambda * (1 - 0.55 * bz)
            elif self.gs.IschemiaDeg == 15:
                Lam_mech = self.params.ekb.llambda * (1 - 0.8 * bz)
            else:
                Lam_mech = self.params.ekb.llambda

            # Решение механической части
            try:
                v_new, l1_new, N_new = solve_mechanical(
                    self.v[i, j], self.l_1[i, j], self.l_2[i, j],
                    self.l_3[i], self.N[i, j], self.Y[i+1, j, :],
                    dt, Lam_mech, self.params
                )

                self.v[i+1, j] = v_new
                self.l_1[i+1, j] = l1_new
                self.N[i+1, j] = N_new
                self.gs.l1_n[j] = l1_new
            except Exception as e:
                print(f"Ошибка в механической части для клетки {j}: {e}")
                self.v[i+1, j] = self.v[i, j]
                self.l_1[i+1, j] = self.l_1[i, j]
                self.N[i+1, j] = self.N[i, j]
                self.gs.l1_n[j] = self.l_1[i, j]

        # Решение для l2 и l3
        self._solve_l2l3(i)

    def _solve_l2l3(self, i):
        """Решение системы для l2 и l3"""
        l2n_l3 = np.concatenate([self.l_2[i, :], [self.l_3[i]]])

        def l2l3_func(x):
            return l2l3(x, self.gs.l1_n, self.gs.L,
                        self.params.sim.dx, self.params)

        try:
            l2_l3_sol = fsolve(l2l3_func, l2n_l3, xtol=1e-8)

            for j in range(self.params.sim.n):
                self.l_2[i+1, j] = l2_l3_sol[j]
            self.l_3[i+1] = l2_l3_sol[self.params.sim.n]
        except Exception as e:
            print(f"Ошибка в l2l3: {e}")
            self.l_2[i+1, :] = self.l_2[i, :]
            self.l_3[i+1] = self.l_3[i]

    def run(self, tnnpe_func, diffusion_func):
        """
        Запуск основного цикла моделирования

        Аргументы:
            tnnpe_func: функция TNNPE
            diffusion_func: функция диффузии
        """
        print("Запуск моделирования...")
        start_time = time.time()

        for i in tqdm(range(self.params.sim.s - 1),
                      desc="Моделирование"):
            self.solve_step(i, tnnpe_func, diffusion_func)

        elapsed = time.time() - start_time
        print(f"Моделирование завершено за {elapsed:.2f} сек")

    def get_results(self):
        """Возвращает результаты моделирования"""
        return {
            'time': self.params.sim.t,
            'x': self.params.sim.x,
            'V': self.Y[:, :, 12],
            'Ca_i': self.Y[:, :, 8],
            'Ca_SR': self.Y[:, :, 7],
            'Na_i': self.Y[:, :, 10],
            'K_i': self.Y[:, :, 9],
            'TRPN': self.Y[:, :, 11],
            'N': self.N,
            'v': self.v,
            'l1': self.l_1,
            'l2': self.l_2,
            'l3': self.l_3,
            'cell_currents': self.cell_currents,
            'deltaU': self.deltaU
        }