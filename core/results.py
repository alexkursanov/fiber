"""
Структуры данных для результатов моделирования
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class SimulationResults:
    """Результаты симуляции"""
    time: np.ndarray
    x: np.ndarray
    
    # Электрические переменные
    V: np.ndarray           # (s, n) - мембранный потенциал, мВ
    Ca_i: np.ndarray        # (s, n) - внутриклеточный кальций, мМ
    Ca_SR: np.ndarray       # (s, n) - кальций в SR, мМ
    Na_i: np.ndarray        # (s, n) - внутриклеточный натрий, мМ
    K_i: np.ndarray         # (s, n) - внутриклеточный калий, мМ
    TRPN: np.ndarray        # (s, n) - тропонин-Ca, мМ
    
    # Механические переменные
    N: np.ndarray           # (s, n) - вероятность поперечных мостиков
    v: np.ndarray           # (s, n) - скорость сокращения
    l1: np.ndarray          # (s, n) - длина сократительного элемента
    l2: np.ndarray          # (s, n) - длина последовательного элемента
    l3: np.ndarray          # (s,) - длина параллельного элемента
    
    # Токи
    cell_currents: np.ndarray  # (s, n, 14) - ионные токи
    
    # Диффузия
    deltaU: np.ndarray      # (s, n) - разность потенциалов между клетками
    
    # Метаданные
    ischemia: int = 0
    duration: float = 0.0
    n_cells: int = 0
    n_time_points: int = 0
    diffusion: float = 0.0
    
    @property
    def n_times(self) -> int:
        return len(self.time)
    
    @property
    def num_cells(self) -> int:
        return len(self.x)
    
    def to_dict(self) -> dict:
        """Конвертация в словарь для обратной совместимости"""
        return {
            'time': self.time,
            'x': self.x,
            'V': self.V,
            'Ca_i': self.Ca_i,
            'Ca_SR': self.Ca_SR,
            'Na_i': self.Na_i,
            'K_i': self.K_i,
            'TRPN': self.TRPN,
            'N': self.N,
            'v': self.v,
            'l1': self.l1,
            'l2': self.l2,
            'l3': self.l3,
            'cell_currents': self.cell_currents,
            'deltaU': self.deltaU,
        }
    
    @classmethod
    def from_dict(cls, data: dict, metadata: Optional[dict] = None) -> 'SimulationResults':
        """Создание из словаря"""
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        
        if metadata:
            kwargs['ischemia'] = int(metadata.get('ischemia', 0))
            kwargs['duration'] = float(metadata.get('duration', 0))
            kwargs['n_cells'] = int(metadata.get('cells', 0))
            kwargs['n_time_points'] = int(metadata.get('time_points', 0))
            kwargs['diffusion'] = float(metadata.get('diffusion', 0))
        
        return cls(**kwargs)


class CurrentIndices:
    """Индексы токов в массиве cell_currents"""
    I_NA = 0      # Натриевый ток
    I_T = 1       # Транзиентный калиевый ток
    I_SS = 2      # Ток установившегося состояния
    I_F = 3       # Funny ток
    I_K1 = 4      # Внутренний выпрямляющий калий
    I_B_NA = 5    # Фоновый натриевый ток
    I_B_K = 6     # Фоновый калиевый ток
    I_NAK = 7     # Na-K насос
    I_STIM = 8    # Ток стимуляции
    I_CA_B = 9    # Кальциевый фоновый ток
    I_NACA = 10   # Na-Ca обменник
    I_PCA = 11    # Ca насос
    I_LCC = 12    # L-тип Ca канал
    I_K_ATP = 13  # K-ATP канал
    
    @classmethod
    def names(cls) -> list:
        return [
            'i_Na', 'i_t', 'i_ss', 'i_f', 'i_K1',
            'i_B_Na', 'i_B_K', 'i_NaK', 'I_Stim',
            'I_CaB', 'I_NaCa', 'I_pCa', 'I_LCC', 'i_K_ATP'
        ]
    
    @classmethod
    def count(cls) -> int:
        return 14


class StateIndices:
    """Индексы переменных состояния в векторе Y"""
    # CaRU состояния
    Z_1 = 0
    Z_2 = 1
    Z_3 = 2
    
    # Калиевые ворота
    R1 = 3
    S = 4
    S_SLOW = 5
    
    # Funny ток ворота
    Y = 6
    
    # Концентрации
    CA_SR = 7
    CA_I = 8
    K_I = 9
    NA_I = 10
    TRPN = 11
    
    # Мембранный потенциал
    V = 12
    
    # Неиспользуемые
    Q_1 = 13
    Q_2 = 14
    Q_3 = 15
    Z = 16
    
    # Na канал ворота
    H = 17
    J = 18
    M = 19
    
    # SS каналы
    R_SS = 20
    S_SS = 21
    
    # Буферы
    B1 = 22
    B2 = 23
    
    @classmethod
    def count(cls) -> int:
        return 24
    
    @classmethod
    def active(cls) -> list:
        """Активные индексы"""
        return [
            cls.Z_1, cls.Z_2, cls.Z_3,
            cls.R1, cls.S, cls.S_SLOW, cls.Y,
            cls.CA_SR, cls.CA_I, cls.K_I, cls.NA_I, cls.TRPN,
            cls.V, cls.H, cls.J, cls.M, cls.R_SS, cls.S_SS
        ]
