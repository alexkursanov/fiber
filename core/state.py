"""
Глобальное состояние модели для передачи между функциями
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GlobalState:
    """Глобальное состояние модели для передачи между функциями"""

    def __init__(self, params=None):
        self.jj = 0                     # индекс текущей клетки
        self.cell_cur = np.zeros(14)    # токи текущей клетки
        self.N_elec = 0.0               # N для электрической части
        self.N_meh = 0.0                # N для механической части
        self.n = 120                     # количество клеток
        self.L = 0.0                     # общая длина
        self.dx = 0.0                    # шаг по пространству

        # Механические переменные
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.l1_n = np.zeros(120)        # l1 для всех клеток

        # Параметры ишемии
        self.IschemiaDeg = 15
        self.BZ1Start = 25
        self.BZ1End = 45
        self.BZ2Start = 75
        self.BZ2End = 95

        # Механическое нагружение
        self.Lam_mech = 55.0

        if params is not None:
            self.update_from_params(params)

    def update_from_params(self, params):
        """Обновляет состояние из параметров моделирования"""
        self.n = params.sim.n
        self.IschemiaDeg = params.sim.IschemiaDeg
        self.BZ1Start = params.sim.BZ1Start
        self.BZ1End = params.sim.BZ1End
        self.BZ2Start = params.sim.BZ2Start
        self.BZ2End = params.sim.BZ2End
        self.dx = params.sim.dx
        self.l1_n = np.zeros(self.n)

    def _calc_bzdegree(self, jj: int) -> float:
        """Вычисляет степень ишемии для клетки с индексом jj (1-based)"""
        if jj < (self.BZ1Start + self.BZ2End) / 2:
            bzdegree = (jj - self.BZ1Start) / (self.BZ1End - self.BZ1Start + 1e-12)
        else:
            bzdegree = (self.BZ2End - jj) / (self.BZ2End - self.BZ2Start + 1e-12)

        return max(0.0, min(1.0, bzdegree))

    def get_bzdegree_for_cell(self, jj: int) -> float:
        """Публичный метод получения степени ишемии"""
        return self._calc_bzdegree(jj)