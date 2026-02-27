"""
Анимация результатов моделирования
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Проверяем наличие IPython для отображения в Jupyter
try:
    from IPython.display import HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class ResultsAnimator:
    """Класс для создания анимаций"""

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

        # Настройка шрифта
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # Определяем шаг для анимации (каждый 10-й кадр для ускорения)
        self.step = max(1, len(self.time) // 500)
        self.time_anim = self.time[::self.step]

    def animate_potential(self, save_path=None, fps=30):
        """Анимация распространения потенциала"""
        fig, ax = plt.subplots(figsize=(10, 6))

        V = self.results['V'][::self.step, :]
        line, = ax.plot([], [], 'b-', linewidth=2)

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(-100, 50)
        ax.set_xlabel('Позиция')
        ax.set_ylabel('Потенциал (мВ)')
        ax.set_title(f'Распространение потенциала, t = {self.time_anim[0]:.1f} мс')
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def update(frame):
            line.set_data(self.x, V[frame, :])
            time_text.set_text(f't = {self.time_anim[frame]:.1f} мс')
            return line, time_text

        anim = FuncAnimation(fig, update, frames=len(V),
                            init_func=init, blit=True, interval=1000/fps)

        if save_path:
            # Всегда сохраняем как GIF
            if not save_path.endswith('.gif'):
                save_path = save_path.replace('.mp4', '.gif')
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print(f"Анимация сохранена как GIF: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить анимацию: {e}")

        plt.close()
        return anim

    def animate_calcium(self, save_path=None, fps=30):
        """Анимация распространения кальция"""
        fig, ax = plt.subplots(figsize=(10, 6))

        Ca = self.results['Ca_i'][::self.step, :] * 1000
        line, = ax.plot([], [], 'r-', linewidth=2)

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, np.max(Ca) * 1.1)
        ax.set_xlabel('Позиция')
        ax.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax.set_title(f'Распространение кальция, t = {self.time_anim[0]:.1f} мс')
        ax.grid(True, alpha=0.3)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def update(frame):
            line.set_data(self.x, Ca[frame, :])
            time_text.set_text(f't = {self.time_anim[frame]:.1f} мс')
            return line, time_text

        anim = FuncAnimation(fig, update, frames=len(Ca),
                            init_func=init, blit=True, interval=1000/fps)

        if save_path:
            # Всегда сохраняем как GIF
            if not save_path.endswith('.gif'):
                save_path = save_path.replace('.mp4', '.gif')
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print(f"Анимация сохранена как GIF: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить анимацию: {e}")

        plt.close()
        return anim

    def animate_dual(self, save_path=None, fps=30):
        """Двойная анимация: потенциал и кальций"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        V = self.results['V'][::self.step, :]
        Ca = self.results['Ca_i'][::self.step, :] * 1000

        line1, = ax1.plot([], [], 'b-', linewidth=2)
        line2, = ax2.plot([], [], 'r-', linewidth=2)

        ax1.set_xlim(self.x[0], self.x[-1])
        ax1.set_ylim(-100, 50)
        ax1.set_ylabel('Потенциал (мВ)')
        ax1.set_title('Распространение потенциала и кальция')
        ax1.grid(True, alpha=0.3)

        ax2.set_xlim(self.x[0], self.x[-1])
        ax2.set_ylim(0, np.max(Ca) * 1.1)
        ax2.set_xlabel('Позиция')
        ax2.set_ylabel('[Ca]i (нМ)')  # Убрали спецсимволы
        ax2.grid(True, alpha=0.3)

        time_text = fig.text(0.02, 0.95, '', fontsize=12)

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            time_text.set_text('')
            return line1, line2, time_text

        def update(frame):
            line1.set_data(self.x, V[frame, :])
            line2.set_data(self.x, Ca[frame, :])
            time_text.set_text(f't = {self.time_anim[frame]:.1f} мс')
            return line1, line2, time_text

        anim = FuncAnimation(fig, update, frames=len(V),
                            init_func=init, blit=True, interval=1000/fps)

        if save_path:
            # Всегда сохраняем как GIF
            if not save_path.endswith('.gif'):
                save_path = save_path.replace('.mp4', '.gif')
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer)
                print(f"Анимация сохранена как GIF: {save_path}")
            except Exception as e:
                print(f"Не удалось сохранить анимацию: {e}")

        plt.tight_layout()
        plt.close()
        return anim

    def to_html(self, anim):
        """Конвертирует анимацию в HTML для отображения в Jupyter"""
        if IPYTHON_AVAILABLE:
            return HTML(anim.to_jshtml())
        else:
            print("IPython не установлен. Невозможно создать HTML.")
            return None

    def show_in_notebook(self, anim_type='dual'):
        """Показывает анимацию в Jupyter notebook"""
        if not IPYTHON_AVAILABLE:
            print("IPython не установлен. Запустите 'pip install ipython' для работы в Jupyter.")
            return None

        if anim_type == 'potential':
            anim = self.animate_potential()
        elif anim_type == 'calcium':
            anim = self.animate_calcium()
        else:
            anim = self.animate_dual()

        return self.to_html(anim)