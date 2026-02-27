# !/usr/bin/env python3
"""
Скрипт для визуализации результатов моделирования без повторного расчета
"""
import argparse
import os
import sys
from datetime import datetime
import numpy as np

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_io.h5_io import load_results_h5
from visualization.plots import ResultsPlotter
from visualization.animation import ResultsAnimator


class SimpleParams:
    """Простой класс-заглушка для параметров"""

    def __init__(self, time, x):
        self.time = time
        self.x = x

    class Sim:
        def __init__(self, time, x):
            self.t = time
            self.x = x

    @property
    def sim(self):
        return self.Sim(self.time, self.x)


def main():
    parser = argparse.ArgumentParser(description='Plot cardiac model results')
    parser.add_argument('input', type=str, help='Input HDF5 file with results')
    parser.add_argument('--cell', type=int, default=0,
                        help='Cell index for single-cell plots')
    parser.add_argument('--time-idx', type=int, default=-1,
                        help='Time index for snapshot (-1 for last)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--fps', type=int, default=15,
                        help='FPS for animations')

    # Выбор графиков
    parser.add_argument('--all', action='store_true',
                        help='Plot all graphs')
    parser.add_argument('--action-potential', action='store_true',
                        help='Plot action potential')
    parser.add_argument('--calcium', action='store_true',
                        help='Plot calcium transient')
    parser.add_argument('--mechanical', action='store_true',
                        help='Plot mechanical variables')
    parser.add_argument('--ions', action='store_true',
                        help='Plot ion concentrations')
    parser.add_argument('--phase', action='store_true',
                        help='Plot phase portrait')
    parser.add_argument('--spatiotemporal', action='store_true',
                        help='Plot spatiotemporal diagram')
    parser.add_argument('--snapshot', action='store_true',
                        help='Plot snapshot of all cells')
    parser.add_argument('--dashboard', action='store_true',
                        help='Plot dashboard')
    parser.add_argument('--animations', action='store_true',
                        help='Create animations')

    # Если ничего не выбрано, рисуем всё
    args = parser.parse_args()

    if not (args.all or args.action_potential or args.calcium or
            args.mechanical or args.ions or args.phase or
            args.spatiotemporal or args.snapshot or args.dashboard or
            args.animations):
        args.all = True

    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ МОДЕЛИРОВАНИЯ")
    print("=" * 60)

    # Загружаем результаты
    print(f"\nЗагрузка результатов из {args.input}...")
    try:
        results, metadata = load_results_h5(args.input)
        print("Результаты успешно загружены")
        if metadata:
            print("Метаданные:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")
        return

    # Создаем простые параметры для визуализации
    time = results.get('time')
    x = results.get('x')

    if time is None:
        print("Ошибка: в файле отсутствует массив 'time'")
        # Создаем временную шкалу на основе размера данных
        V = results.get('V')
        if V is not None:
            time = np.arange(V.shape[0])
            print(f"Создана временная шкала от 0 до {V.shape[0] - 1}")

    if x is None:
        print("Ошибка: в файле отсутствует массив 'x'")
        V = results.get('V')
        if V is not None and len(V.shape) > 1:
            x = np.arange(V.shape[1])
            print(f"Создана пространственная шкала от 0 до {V.shape[1] - 1}")

    # Создаем класс с параметрами
    class PlotParams:
        def __init__(self, time, x):
            self.time = time
            self.x = x

        class Sim:
            def __init__(self, time, x):
                self.t = time
                self.x = x

        @property
        def sim(self):
            return self.Sim(self.time, self.x)

    params = PlotParams(time, x)

    # Создание папки для графиков
    if args.output_dir is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        plots_dir = f"{base_name}_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        plots_dir = args.output_dir

    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nГрафики сохраняются в: {plots_dir}")

    # Визуализация
    plotter = ResultsPlotter(results, params)

    # Базовые графики
    if args.all or args.action_potential:
        print("   - Потенциал действия...")
        try:
            plotter.plot_action_potential(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '01_action_potential.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.calcium:
        print("   - Кальциевый транзиент...")
        try:
            plotter.plot_calcium_transient(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '02_calcium.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.mechanical:
        print("   - Механические переменные...")
        try:
            plotter.plot_mechanical_variables(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '03_mechanical.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.ions:
        print("   - Ионные концентрации...")
        try:
            plotter.plot_ions_concentrations(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '04_ions.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.phase:
        print("   - Фазовый портрет...")
        try:
            plotter.plot_phase_portrait(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '05_phase_portrait.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.spatiotemporal:
        print("   - Пространственно-временная диаграмма...")
        try:
            plotter.plot_spatiotemporal(
                save_path=os.path.join(plots_dir, '06_spatiotemporal.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.snapshot:
        print("   - Срез по всем клеткам...")
        try:
            plotter.plot_all_cells_snapshot(
                time_idx=args.time_idx,
                save_path=os.path.join(plots_dir, '07_snapshot.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    if args.all or args.dashboard:
        print("   - Дашборд...")
        try:
            plotter.create_dashboard(
                cell_idx=args.cell,
                save_path=os.path.join(plots_dir, '08_dashboard.png'))
        except Exception as e:
            print(f"   ! Ошибка: {e}")

    # Анимации
    if args.animations or args.all:
        print("\nСоздание анимаций...")
        try:
            animator = ResultsAnimator(results, params)

            if args.animations or args.all:
                print("   - Анимация потенциала...")
                anim_path = os.path.join(plots_dir, 'animation_potential.gif')
                animator.animate_potential(save_path=anim_path, fps=args.fps)

            if args.animations or args.all:
                print("   - Анимация кальция...")
                anim_path = os.path.join(plots_dir, 'animation_calcium.gif')
                animator.animate_calcium(save_path=anim_path, fps=args.fps)

            if args.animations or args.all:
                print("   - Двойная анимация...")
                anim_path = os.path.join(plots_dir, 'animation_dual.gif')
                animator.animate_dual(save_path=anim_path, fps=args.fps)

        except Exception as e:
            print(f"   ! Не удалось создать анимации: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nГрафики сохранены в папку {plots_dir}")
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
    print("=" * 60)


if __name__ == "__main__":
    main()