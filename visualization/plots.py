"""
Функции для построения графиков результатов моделирования
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ResultsPlotter:
    """Класс для визуализации результатов"""

    def __init__(self, results, params):
        """
        Инициализация

        Аргументы:
            results: словарь с результатами
            params: параметры модели
        """
        self.results = results
        self.params = params
        self.time = results['time']
        self.x = results['x']

        # Настройка стиля - используем шрифт с поддержкой Unicode
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Шрифт с поддержкой Unicode

    def plot_action_potential(self, cell_idx=0, save_path=None):
        """График потенциала действия для одной клетки"""
        fig, ax = plt.subplots(figsize=(10, 6))

        V = self.results['V'][:, cell_idx]
        ax.plot(self.time, V, 'b-', linewidth=2)

        ax.set_xlabel('Время (мс)')
        ax.set_ylabel('Потенциал (мВ)')
        ax.set_title(f'Потенциал действия (клетка {cell_idx})')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_calcium_transient(self, cell_idx=0, save_path=None):
        """График кальциевого транзиента"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        Ca_i = self.results['Ca_i'][:, cell_idx] * 1000  # в nM
        Ca_SR = self.results['Ca_SR'][:, cell_idx]

        ax1.plot(self.time, Ca_i, 'r-', linewidth=2)
        ax1.set_xlabel('Время (мс)')
        ax1.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax1.set_title(f'Внутриклеточный кальций (клетка {cell_idx})')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.time, Ca_SR, 'g-', linewidth=2)
        ax2.set_xlabel('Время (мс)')
        ax2.set_ylabel('[Ca]SR (мМ)')  # Убрали спецсимволы
        ax2.set_title('Кальций в саркоплазматическом ретикулуме')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_mechanical_variables(self, cell_idx=0, save_path=None):
        """График механических переменных"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

        v = self.results['v'][:, cell_idx] * 1000  # в нм/мс
        N = self.results['N'][:, cell_idx]
        l1 = self.results['l1'][:, cell_idx]

        ax1.plot(self.time, v, 'b-', linewidth=2)
        ax1.set_xlabel('Время (мс)')
        ax1.set_ylabel('Скорость (нм/мс)')
        ax1.set_title(f'Скорость сокращения (клетка {cell_idx})')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.time, N, 'r-', linewidth=2)
        ax2.set_xlabel('Время (мс)')
        ax2.set_ylabel('N')
        ax2.set_title('Вероятность прикрепления мостиков')
        ax2.grid(True, alpha=0.3)

        ax3.plot(self.time, l1, 'g-', linewidth=2)
        ax3.set_xlabel('Время (мс)')
        ax3.set_ylabel('l1')  # Убрали индекс
        ax3.set_title('Длина элемента 1')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_phase_portrait(self, cell_idx=0, save_path=None):
        """Фазовый портрет (V vs Ca)"""
        fig, ax = plt.subplots(figsize=(8, 8))

        V = self.results['V'][:, cell_idx]
        Ca = self.results['Ca_i'][:, cell_idx] * 1000

        # Цвет по времени
        scatter = ax.scatter(V, Ca, c=self.time, cmap='viridis',
                            s=1, alpha=0.6)
        ax.plot(V, Ca, 'k-', alpha=0.2, linewidth=0.5)

        ax.set_xlabel('Потенциал (мВ)')
        ax.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax.set_title(f'Фазовый портрет (клетка {cell_idx})')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, label='Время (мс)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_spatiotemporal(self, save_path=None):
        """Пространственно-временная диаграмма потенциала"""
        fig, ax = plt.subplots(figsize=(12, 8))

        V = self.results['V'].T  # транспонируем для правильной ориентации

        im = ax.imshow(V, aspect='auto', cmap='RdBu_r',
                      extent=[self.time[0], self.time[-1],
                              self.x[-1], self.x[0]],
                      vmin=-100, vmax=50)

        ax.set_xlabel('Время (мс)')
        ax.set_ylabel('Позиция')
        ax.set_title('Пространственно-временная диаграмма потенциала')

        plt.colorbar(im, label='Потенциал (мВ)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ions_concentrations(self, cell_idx=0, save_path=None):
        """График ионных концентраций"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        Na = self.results['Na_i'][:, cell_idx]
        K = self.results['K_i'][:, cell_idx]

        ax1.plot(self.time, Na, 'b-', linewidth=2)
        ax1.set_xlabel('Время (мс)')
        ax1.set_ylabel('[Na]i (мМ)')  # Убрали спецсимволы
        ax1.set_title(f'Внутриклеточный натрий (клетка {cell_idx})')
        ax1.grid(True, alpha=0.3)

        ax2.plot(self.time, K, 'r-', linewidth=2)
        ax2.set_xlabel('Время (мс)')
        ax2.set_ylabel('[K]i (мМ)')  # Убрали спецсимволы
        ax2.set_title('Внутриклеточный калий')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_cells_snapshot(self, time_idx=-1, save_path=None):
        """Срез по всем клеткам в заданный момент времени"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        V = self.results['V'][time_idx, :]
        Ca = self.results['Ca_i'][time_idx, :] * 1000
        N = self.results['N'][time_idx, :]
        l1 = self.results['l1'][time_idx, :]

        axes[0, 0].plot(self.x, V, 'b-o', markersize=3)
        axes[0, 0].set_xlabel('Позиция')
        axes[0, 0].set_ylabel('Потенциал (мВ)')
        axes[0, 0].set_title(f'Потенциал в t={self.time[time_idx]:.1f} мс')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(self.x, Ca, 'r-o', markersize=3)
        axes[0, 1].set_xlabel('Позиция')
        axes[0, 1].set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        axes[0, 1].set_title('Внутриклеточный кальций')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(self.x, N, 'g-o', markersize=3)
        axes[1, 0].set_xlabel('Позиция')
        axes[1, 0].set_ylabel('N')
        axes[1, 0].set_title('Вероятность прикрепления мостиков')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(self.x, l1, 'm-o', markersize=3)
        axes[1, 1].set_xlabel('Позиция')
        axes[1, 1].set_ylabel('l1')  # Убрали индекс
        axes[1, 1].set_title('Длина элемента 1')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_dashboard(self, cell_idx=0, save_path=None):
        """Создает дашборд с основными графиками"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)

        # Потенциал действия
        ax1 = fig.add_subplot(gs[0, :2])
        V = self.results['V'][:, cell_idx]
        ax1.plot(self.time, V, 'b-', linewidth=2)
        ax1.set_xlabel('Время (мс)')
        ax1.set_ylabel('Потенциал (мВ)')
        ax1.set_title('Потенциал действия')
        ax1.grid(True, alpha=0.3)

        # Кальциевый транзиент
        ax2 = fig.add_subplot(gs[0, 2])
        Ca = self.results['Ca_i'][:, cell_idx] * 1000
        ax2.plot(self.time, Ca, 'r-', linewidth=2)
        ax2.set_xlabel('Время (мс)')
        ax2.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax2.set_title('Кальциевый транзиент')
        ax2.grid(True, alpha=0.3)

        # Скорость сокращения
        ax3 = fig.add_subplot(gs[1, 0])
        v = self.results['v'][:, cell_idx] * 1000
        ax3.plot(self.time, v, 'g-', linewidth=2)
        ax3.set_xlabel('Время (мс)')
        ax3.set_ylabel('Скорость (нм/мс)')
        ax3.set_title('Скорость сокращения')
        ax3.grid(True, alpha=0.3)

        # N
        ax4 = fig.add_subplot(gs[1, 1])
        N = self.results['N'][:, cell_idx]
        ax4.plot(self.time, N, 'm-', linewidth=2)
        ax4.set_xlabel('Время (мс)')
        ax4.set_ylabel('N')
        ax4.set_title('Вероятность прикрепления мостиков')
        ax4.grid(True, alpha=0.3)

        # Фазовый портрет
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.scatter(V, Ca, c=self.time, cmap='viridis', s=1, alpha=0.6)
        ax5.set_xlabel('Потенциал (мВ)')
        ax5.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax5.set_title('Фазовый портрет')
        ax5.grid(True, alpha=0.3)

        # Пространственно-временная диаграмма (уменьшенная)
        ax6 = fig.add_subplot(gs[2, :])
        V_all = self.results['V'].T
        im = ax6.imshow(V_all, aspect='auto', cmap='RdBu_r',
                       extent=[self.time[0], self.time[-1],
                               self.x[-1], self.x[0]],
                       vmin=-100, vmax=50)
        ax6.set_xlabel('Время (мс)')
        ax6.set_ylabel('Позиция')
        ax6.set_title('Пространственно-временная диаграмма')
        plt.colorbar(im, ax=ax6, label='Потенциал (мВ)')

        plt.suptitle(f'Дашборд моделирования (клетка {cell_idx})', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()