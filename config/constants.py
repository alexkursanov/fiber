"""
Константы и магические числа модели
"""


class ElectricalConstants:
    """Физические константы"""
    F = 96487.0      # Постоянная Фарадея, C/mol
    R = 8314.5       # Газовая постоянная, mJ/mol·K
    TEM = 295.0       # Температура, K


class IonConcentrations:
    """Концентрации ионов"""
    CA_O = 1.2       # Внеклеточный кальций, mM
    K_O = 5.4        # Внеклеточный калий, mM
    NA_O = 140.0     # Внеклеточный натрий, mM
    ATP_I = 6.8      # Внутриклеточный ATP, mM


class CurrentNames:
    """Имена токов для визуализации"""
    I_NA = 'Na+ ток'
    I_T = 'Транзиентный K'
    I_SS = 'Установившийся K'
    I_F = 'Funny ток'
    I_K1 = 'Внутренний K'
    I_B_NA = 'Фоновый Na'
    I_B_K = 'Фоновый K'
    I_NAK = 'Na/K насос'
    I_STIM = 'Стимул'
    I_CA_B = 'Ca фон'
    I_NACA = 'Na/Ca обмен'
    I_PCA = 'Ca насос'
    I_LCC = 'L-тип Ca'
    I_K_ATP = 'K-ATP'


class StateVariableNames:
    """Имена переменных состояния"""
    Z_1 = 'z1 CaRU'
    Z_2 = 'z2 CaRU'
    Z_3 = 'z3 CaRU'
    R1 = 'r1 ворота'
    S = 's ворота'
    S_SLOW = 's_slow ворота'
    Y = 'y ворота'
    CA_SR = 'Ca в SR'
    CA_I = 'Внутриклеточный Ca'
    K_I = 'Внутриклеточный K'
    NA_I = 'Внутриклеточный Na'
    TRPN = 'Тропонин-Ca'
    V = 'Потенциал'
    H = 'h ворота'
    J = 'j ворота'
    M = 'm ворота'
    R_SS = 'r_ss ворота'
    S_SS = 's_ss ворота'


class IschemiaDefaults:
    """Значения по умолчанию для ишемии"""
    DEGREE_0 = 0
    DEGREE_5 = 5
    DEGREE_10 = 10
    DEGREE_15 = 15
    
    BZ1_START = 25
    BZ1_END = 45
    BZ2_START = 75
    BZ2_END = 95
    
    # Факторы снижения для каждой степени ишемии
    ATP_FACTORS = {
        5: 0.2,
        10: 0.37,
        15: 0.53
    }
    
    K_O_FACTORS = {
        5: 1.885,
        10: 1.885,
        15: 4.5
    }
    
    G_NA_FACTORS = {
        5: 0.125,
        10: 0.25,
        15: 0.375
    }
    
    J_L_FACTORS = {
        5: 0.15,
        10: 0.30,
        15: 0.70
    }
    
    # Факторы для механики
    LAM_FACTORS = {
        5: 0.43636,
        10: 0.55,
        15: 0.80
    }


class SimulationDefaults:
    """Параметры симуляции по умолчанию"""
    DURATION_MS = 1000.0    # мс
    TIME_POINTS = 1000
    N_CELLS = 80
    DIFFUSION = 150.0
    DTKANI_MM = 20.0        # мм


class NumericalTolerances:
    """Допуски для численных методов"""
    RTOL_BDF = 1e-6
    ATOL_BDF = 1e-8
    
    RTOL_RELAXED = 1e-4
    ATOL_RELAXED = 1e-5
    
    XTOL_FSOLVE = 1e-8
    XTOL_FSOLVE_RELAXED = 1e-4


class StateIndex:
    """Индексы в векторе состояния Y"""
    # CaRU состояния
    Z_1 = 0
    Z_2 = 1
    Z_3 = 2
    
    # Калиевые ворота
    R1 = 3
    S = 4
    S_SLOW = 5
    
    # Funny ток
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
    
    N_STATES = 24


class CurrentIndex:
    """Индексы в массиве токов"""
    I_NA = 0
    I_T = 1
    I_SS = 2
    I_F = 3
    I_K1 = 4
    I_B_NA = 5
    I_B_K = 6
    I_NAK = 7
    I_STIM = 8
    I_CA_B = 9
    I_NACA = 10
    I_PCA = 11
    I_LCC = 12
    I_K_ATP = 13
    
    N_CURRENTS = 14
