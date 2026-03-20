"""
Параметры модели электро-механического сопряжения кардиомиоцита
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional


@dataclass
class EKBParameters:
    """Параметры модели EKB (Izakov et al.) - механическая часть"""

    # Параметры пассивных элементов
    alpha_1: float = 21.0      # per_micrometre
    beta_1: float = 0.94        # millinewton
    alpha_2: float = 14.6       # per_micrometre
    beta_2: float = 0.0018      # millinewton
    alpha_3: float = 33.79      # per_micrometre
    beta_3: float = 0.0084      # millinewton

    # Кинетические параметры
    q_1: float = 0.0173         # per_second
    q_2: float = 0.259          # per_second
    q_3: float = 0.0173         # per_second
    q_4: float = 0.015          # per_second

    v_max: float = 0.0055       # micrometre_per_second
    a: float = 0.25             # dimensionless
    alpha_Q: float = 10.0       # dimensionless
    beta_Q: float = 5.0         # dimensionless

    x_st: float = 0.964285      # dimensionless
    alpha_G: float = 1.0        # dimensionless

    m_0: float = 0.9            # dimensionless
    g_1: float = 0.6            # per_micrometre
    g_2: float = 0.52           # dimensionless

    S_0: float = 1.14           # micrometre
    chi_1: float = 0.55         # dimensionless
    chi_2: float = 0.0          # dimensionless
    r0: float = 0.12            # preload

    d_h: float = 0.5            # dimensionless
    m: float = 1.7              # dimensionless
    chi_0: float = 2.1          # dimensionless
    alpha_P: float = 4.0        # dimensionless
    q_st: float = 1000.0        # dimensionless

    # Вязкие параметры
    alpha_vp_l: float = 16.0    # per_micrometre
    alpha_vp_s: float = 8.0     # per_micrometre
    beta_vp_l: float = 0.00084 * 0.9  # millinewton_second_per_micrometre
    beta_vp_s: float = 0.84     # millinewton_second_per_micrometre

    # Кальциевая кинетика
    a_off: float = 0.17         # per_second
    a_on: float = 35.0          # per_second
    B_1_tot: float = 0.0        # millimolar
    B_2_tot: float = 0.0        # millimolar
    a_eqmin: float = 0.001299042
    tau_inf: float = 1500.0

    # Параметры для n1
    k_mu: float = 0.6           # dimensionless
    mu: float = 3.3             # dimensionless
    s_c: float = 1.0

    # Параметры для тропонина
    k_A: float = 28.0           # per_millimolar
    A_tot: float = 0.07         # millimolar

    # Параметры для n1
    n1_A: float = 0.5
    n1_B: float = 55.0
    n1_C: float = 1.0
    n1_Q: float = 0.835
    n1_K: float = 1.0
    n1_nu: float = 5.0

    # Геометрические параметры
    s055: float = 0.55
    s046: float = 0.46

    # Параметры ткани
    llambda: float = 55.0       # millinewton

    # Флаги
    nondimension: bool = True
    L_0: float = 1.67
    R_0: float = 1.05

    def apply_nondim(self):
        """Применяет обезразмеривание если необходимо"""
        if self.nondimension:
            restlength = self.L_0
            self.alpha_1 *= restlength
            self.alpha_2 *= restlength
            self.alpha_3 *= restlength
            self.beta_Q *= restlength
            self.alpha_vp_l *= restlength
            self.beta_vp_l *= restlength
            self.beta_vp_s *= restlength
            self.g_1 *= restlength
            self.n1_B *= restlength
            self.v_max /= restlength
            self.S_0 /= restlength
            self.s055 /= restlength
            self.s046 /= restlength


@dataclass
class ElectricalParameters:
    """Параметры электрофизиологической модели (TNNP / Hinch)"""

    # Ca handling
    GeneralTnC: float = 0.07
    BindingConstTnC: float = 1000.0
    C20: float = 8.0
    qa: float = 35.0

    # CaRU parameters
    K_L: float = 0.00022
    K_RyR: float = 0.041
    V_L: float = -2.0
    a1: float = 0.0625
    b: float = 14.0
    c: float = 0.01
    TNP_d: float = 100.0
    del_VL: float = 7.0
    phi_L: float = 2.35
    phi_R: float = 0.05
    t_L: float = 1.0
    tau_L: float = 650.0
    tau_R_1: float = 2.43
    theta_R: float = 0.012

    # Cell geometry
    V_SR_uL: float = 2.098e-6
    V_myo: float = 25850.0
    V_myo_uL: float = 2.585e-5

    # Channel conductances
    g_CaB: float = 2.6875e-8
    J_L: float = 0.000913
    J_R: float = 0.02
    N_L_CaRU: int = 50000
    N_R_CaRU: int = 50000
    g_D: float = 0.065

    # Na-Ca exchanger
    K_mCa: float = 1.38
    K_mNa: float = 87.5
    eta: float = 0.35
    g_NCX: float = 0.0385
    k_sat: float = 0.1

    # SERCA
    K_SERCA: float = 0.0005
    g_SERCA: float = 0.00045

    # SR leak
    g_SRl: float = 1.8951e-5

    # Sarcolemmal Ca pump
    K_mpCa: float = 0.0005
    g_pCa: float = 3.50e-06

    # Buffers
    B_CMDN: float = 0.05
    k_CMDN: float = 0.002382

    # Membrane
    Cm: float = 0.0001  # uF
    F: float = 96487.0  # C/mol
    R: float = 8314.5   # mJ/mol·K
    Tem: float = 295.0  # K

    # Stimulation
    stim_amplitude: float = -0.0006  # uA
    stim_duration: float = 10.0      # ms
    stim_period: float = 1000.0      # ms
    StimStart_shift: float = 60.0

    # Ion channel parameters (Gattoni / Pandit)
    a_to: float = 0.883
    b_to: float = 0.117
    g_t: float = 1.96e-05
    g_B_K: float = 1.38e-07
    g_B_Na: float = 8.015e-8
    f_Na: float = 0.2
    g_f: float = 1.45e-06
    g_K1: float = 2.40e-05
    g_Na: float = 0.0008
    K_m_K: float = 1.5
    K_m_Na: float = 10.0
    i_NaK_max: float = 9.50e-05
    g_ss: float = 7.00e-06

    # K_ATP channel
    g_K_ATP: float = 1.150e-3
    V50_P_ATP: float = 0.1
    hill_P_ATP: float = 2.0

    # Ionic concentrations
    Ca_o: float = 1.2
    K_o: float = 5.4
    Na_o: float = 140.0
    ATP_i: float = 6.8

    # Ischemia effect on K1
    deg_Ko_K1: float = 0.5

    @property
    def K_o_norm(self) -> float:
        return 5.4

    @property
    def t_R(self) -> float:
        return 1.17 * self.t_L

    @property
    def alpha_m(self) -> float:
        return self.phi_L / self.t_L

    @property
    def beta_m(self) -> float:
        return self.phi_R / self.t_R

    @property
    def g_Na_endo(self) -> float:
        return 1.33 * self.g_Na

    @property
    def sigma(self) -> float:
        return (np.exp(self.Na_o / 67.3) - 1.0) / 7.0

    @property
    def f_K(self) -> float:
        return 1.0 - self.f_Na

    @property
    def tau_s_ss(self) -> float:
        return 2100.0


@dataclass
class SimulationParameters:
    """Параметры моделирования"""

    # Временные параметры (мс)
    t0: float = 0.0
    ts: float = 1000.0
    s: int = 1000  # количество временных точек

    # Пространственные параметры
    x0: float = 0.0
    xn: float = 1.0
    n: int = 80    # количество клеток

    # Диффузия
    D: float = 150.0
    L_tkani: float = 20.0  # mm

    # Ишемия
    IschemiaDeg: int = 15  # 0,5,10,15
    BZ1Start: int = 25
    BZ1End: int = 45
    BZ2Start: int = 75
    BZ2End: int = 95

    # Флаги загрузки
    loadfromfileflag: bool = False
    input_file: str = 'PhV-120-20mm-N-10min(30)-N(GZ20_ro12_D150_LCC)_lam-55.xlsx'

    def __post_init__(self):
        """Вычисляет производные параметры после инициализации"""
        self._compute_derived()

    def __setattr__(self, name, value):
        """Автоматически пересчитывает производные параметры при изменении"""
        super().__setattr__(name, value)
        # После изменения атрибута пересчитываем производные параметры
        # Проверяем, что объект уже инициализирован (есть _compute_derived)
        if hasattr(self, '_compute_derived') and hasattr(self, '_dt'):
            if name in ('t0', 'ts', 's', 'x0', 'xn', 'n', 'D', 'L_tkani'):
                self._compute_derived()

    def _compute_derived(self):
        """Вычисляет производные параметры"""
        self._dt = (self.ts - self.t0) / (self.s - 1) if self.s > 1 else 0
        self._dx = (self.xn - self.x0) / (self.n - 1) if self.n > 1 else 0
        self._D_odez = self.D / (self.L_tkani) ** 2 if self.L_tkani != 0 else 0
        self._t = np.linspace(self.t0, self.ts, self.s)
        self._x = np.linspace(self.x0, self.xn, self.n)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def dx(self) -> float:
        return self._dx

    @property
    def D_odez(self) -> float:
        return self._D_odez

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def x(self) -> np.ndarray:
        return self._x


@dataclass
class ModelParameters:
    """Все параметры модели"""
    ekb: EKBParameters = field(default_factory=EKBParameters)
    elec: ElectricalParameters = field(default_factory=ElectricalParameters)
    sim: SimulationParameters = field(default_factory=SimulationParameters)

    def __post_init__(self):
        """Применяем обезразмеривание после инициализации"""
        self.ekb.apply_nondim()
        self.sim._compute_derived()