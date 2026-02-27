#!/usr/bin/env python3
"""
Главный скрипт для запуска модели электро-механического сопряжения кардиомиоцита
"""
import numpy as np
import argparse
import os
import sys
from datetime import datetime

# Добавляем путь к проекту в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.parameters import ModelParameters, EKBParameters, ElectricalParameters, SimulationParameters
from core.state import GlobalState
from core.solver import CardiacSolver
from models.electrical import tnnpe, y_init
from models.diffusion import solve_diffusion
from file_io.h5_io import save_results_h5, load_results_h5, save_checkpoint, load_checkpoint

# Попытка импорта визуализации
try:
    from visualization.plots import ResultsPlotter
    from visualization.animation import ResultsAnimator

    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"Визуализация недоступна: {e}")


def create_default_params():
    """Создает параметры модели по умолчанию"""
    ekb_params = EKBParameters()
    elec_params = ElectricalParameters()
    sim_params = SimulationParameters()

    return ModelParameters(ekb=ekb_params, elec=elec_params, sim=sim_params)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Cardiac electromechanical model')
    parser.add_argument('--output', type=str, default='results.h5',
                        help='Output file for results')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    parser.add_argument('--ischemia', type=int, default=15,
                        choices=[0, 5, 10, 15],
                        help='Ischemia duration in minutes')
    parser.add_argument('--duration', type=float, default=1000.0,
                        help='Simulation duration in ms')
    parser.add_argument('--cells', type=int, default=80,
                        help='Number of cells')
    parser.add_argument('--time-points', type=int, default=1000,
                        help='Number of time points')
    parser.add_argument('--diffusion', type=float, default=150.0,
                        help='Diffusion coefficient')
    parser.add_argument('--load-init', type=str,
                        help='Load initial conditions from file')

    args = parser.parse_args()

    print("=" * 60)
    print("МОДЕЛЬ ЭЛЕКТРО-МЕХАНИЧЕСКОГО СОПРЯЖЕНИЯ КАРДИОМИОЦИТА")
    print("=" * 60)

    # Создание параметров
    print("\n[1/6] Инициализация параметров...")
    params = create_default_params()
    params.sim.IschemiaDeg = args.ischemia
    params.sim.ts = args.duration
    params.sim.n = args.cells
    params.sim.s = args.time_points
    params.sim.D = args.diffusion

    print(f"   - Ишемия: {args.ischemia} мин")
    print(f"   - Длительность: {args.duration} мс")
    print(f"   - Клеток: {args.cells}")
    print(f"   - Временных точек: {args.time_points}")

    # Глобальное состояние
    print("\n[2/6] Создание глобального состояния...")
    global_state = GlobalState(params)

    # Создание решателя
    print("\n[3/6] Создание решателя...")
    solver = CardiacSolver(params, global_state)

    # Загрузка начальных условий
    print("\n[4/6] Установка начальных условий...")
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"   - Загрузка контрольной точки: {args.checkpoint}")
        start_step = load_checkpoint(args.checkpoint, solver)
        print(f"   - Возобновление с шага {start_step}")
    else:
        print("   - Использование стандартных начальных условий")
        solver.set_initial_conditions(y_init())

    # Запуск моделирования
    print("\n[5/6] Запуск моделирования...")
    try:
        solver.run(tnnpe, solve_diffusion)
    except KeyboardInterrupt:
        print("\nМоделирование прервано пользователем")
        print("Сохранение контрольной точки...")
        checkpoint_file = args.output.replace('.h5', '_interrupted.h5')
        save_checkpoint(checkpoint_file, solver, solver.params.sim.s - 2)
        sys.exit(0)
    except Exception as e:
        print(f"\nОшибка при моделировании: {e}")
        print("Сохранение контрольной точки...")
        checkpoint_file = args.output.replace('.h5', '_error.h5')
        if hasattr(solver, 'Y'):
            current_step = np.where(solver.Y[:, 0, 0] != 0)[0][-1] if np.any(solver.Y[:, 0, 0] != 0) else 0
            save_checkpoint(checkpoint_file, solver, current_step)
        sys.exit(1)

    # Получение результатов
    print("\n[6/6] Обработка результатов...")
    results = solver.get_results()

    # Сохранение результатов
    print("\nСохранение результатов...")
    metadata = {
        'ischemia': args.ischemia,
        'duration': args.duration,
        'cells': args.cells,
        'time_points': args.time_points,
        'diffusion': args.diffusion,
        'timestamp': datetime.now().isoformat()
    }
    save_results_h5(args.output, results, params, metadata)

    # Сохранение контрольной точки
    checkpoint_file = args.output.replace('.h5', '_checkpoint.h5')
    save_checkpoint(checkpoint_file, solver, params.sim.s - 1)

    # Визуализация
    if not args.no_plot and VISUALIZATION_AVAILABLE:
        print("\nСоздание графиков...")
        plotter = ResultsPlotter(results, params)

        # Создание папки для графиков
        plots_dir = f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(plots_dir, exist_ok=True)
        print(f"   - Графики сохраняются в: {plots_dir}")

        try:
            # Базовые графики для первой клетки
            print("   - Потенциал действия...")
            plotter.plot_action_potential(
                save_path=os.path.join(plots_dir, '01_action_potential.png'))

            print("   - Кальциевый транзиент...")
            plotter.plot_calcium_transient(
                save_path=os.path.join(plots_dir, '02_calcium.png'))

            print("   - Механические переменные...")
            plotter.plot_mechanical_variables(
                save_path=os.path.join(plots_dir, '03_mechanical.png'))

            print("   - Пространственно-временная диаграмма...")
            plotter.plot_spatiotemporal(
                save_path=os.path.join(plots_dir, '04_spatiotemporal.png'))

            print("   - Дашборд...")
            plotter.create_dashboard(
                save_path=os.path.join(plots_dir, '05_dashboard.png'))
        except Exception as e:
            print(f"   ! Ошибка при создании графиков: {e}")

            # Анимации
        if not args.no_plot and VISUALIZATION_AVAILABLE:
            print("\nСоздание анимаций...")
            try:
                animator = ResultsAnimator(results, params)

                print("   - Анимация потенциала...")
                anim_path = os.path.join(plots_dir, 'animation_potential.gif')
                animator.animate_potential(save_path=anim_path, fps=15)

                print("   - Анимация кальция...")
                anim_path = os.path.join(plots_dir, 'animation_calcium.gif')
                animator.animate_calcium(save_path=anim_path, fps=15)

                print("   - Двойная анимация...")
                anim_path = os.path.join(plots_dir, 'animation_dual.gif')
                animator.animate_dual(save_path=anim_path, fps=15)

            except Exception as e:
                print(f"   ! Не удалось создать анимации: {e}")
                import traceback
                traceback.print_exc()

        print(f"\nГрафики сохранены в папку {plots_dir}")
    elif not VISUALIZATION_AVAILABLE and not args.no_plot:
        print("\nВизуализация недоступна. Установите matplotlib и seaborn.")

    print("\n" + "=" * 60)
    print("МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("=" * 60)
    print(f"Результаты сохранены в: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
