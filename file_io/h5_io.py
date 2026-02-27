"""
Сохранение и загрузка результатов в формате HDF5
"""
import h5py
import numpy as np
import os
from datetime import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_results_h5(filename, results, params, metadata=None):
    """
    Сохраняет результаты моделирования в HDF5 файл

    Аргументы:
        filename: имя файла
        results: словарь с результатами
        params: параметры модели
        metadata: дополнительная метаинформация
    """
    with h5py.File(filename, 'w') as f:
        # Сохраняем метаданные
        meta_grp = f.create_group('metadata')
        meta_grp.attrs['timestamp'] = datetime.now().isoformat()
        meta_grp.attrs['description'] = 'Cardiac electromechanical model results'

        if metadata:
            for key, value in metadata.items():
                meta_grp.attrs[key] = str(value)

        # Сохраняем параметры
        params_grp = f.create_group('parameters')

        # Параметры EKB
        if hasattr(params, 'ekb'):
            ekb_grp = params_grp.create_group('ekb')
            for key, value in params.ekb.__dict__.items():
                if not key.startswith('_'):
                    try:
                        ekb_grp.attrs[key] = value
                    except:
                        ekb_grp.attrs[key] = str(value)

        # Параметры Electrical
        if hasattr(params, 'elec'):
            elec_grp = params_grp.create_group('electrical')
            for key, value in params.elec.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    try:
                        elec_grp.attrs[key] = value
                    except:
                        elec_grp.attrs[key] = str(value)

        # Параметры симуляции
        if hasattr(params, 'sim'):
            sim_grp = params_grp.create_group('simulation')
            for key, value in params.sim.__dict__.items():
                if not key.startswith('_'):
                    try:
                        sim_grp.attrs[key] = value
                    except:
                        sim_grp.attrs[key] = str(value)

        # Сохраняем результаты
        results_grp = f.create_group('results')

        for name, data in results.items():
            if data is not None:
                # Проверяем, что данные можно сохранить
                if isinstance(data, np.ndarray):
                    results_grp.create_dataset(name, data=data, compression='gzip')
                elif isinstance(data, (int, float, str)):
                    results_grp.attrs[name] = data

    print(f"Результаты сохранены в {filename}")


def load_results_h5(filename):
    """
    Загружает результаты моделирования из HDF5 файла

    Аргументы:
        filename: имя файла

    Возвращает:
        results: словарь с результатами
        metadata: метаданные
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")

    results = {}
    metadata = {}

    with h5py.File(filename, 'r') as f:
        # Загружаем метаданные
        if 'metadata' in f:
            for key, value in f['metadata'].attrs.items():
                metadata[key] = value

        # Загружаем результаты
        if 'results' in f:
            for name in f['results']:
                results[name] = f['results'][name][:]

    print(f"Загружены результаты из {filename}")
    return results, metadata


def save_checkpoint(filename, solver, step):
    """
    Сохраняет контрольную точку для возобновления расчета

    Аргументы:
        filename: имя файла
        solver: объект решателя
        step: текущий шаг
    """
    with h5py.File(filename, 'w') as f:
        # Сохраняем текущий шаг
        f.attrs['current_step'] = step

        # Сохраняем массивы
        if hasattr(solver, 'Y'):
            f.create_dataset('Y', data=solver.Y[:step+1], compression='gzip')
        if hasattr(solver, 'l_1'):
            f.create_dataset('l_1', data=solver.l_1[:step+1], compression='gzip')
        if hasattr(solver, 'l_2'):
            f.create_dataset('l_2', data=solver.l_2[:step+1], compression='gzip')
        if hasattr(solver, 'l_3'):
            f.create_dataset('l_3', data=solver.l_3[:step+1], compression='gzip')
        if hasattr(solver, 'N'):
            f.create_dataset('N', data=solver.N[:step+1], compression='gzip')
        if hasattr(solver, 'v'):
            f.create_dataset('v', data=solver.v[:step+1], compression='gzip')

        if hasattr(solver, 'cell_currents'):
            if solver.cell_currents is not None:
                f.create_dataset('cell_currents',
                               data=solver.cell_currents[:step+1],
                               compression='gzip')

    print(f"Контрольная точка сохранена в {filename} (шаг {step})")


def load_checkpoint(filename, solver):
    """
    Загружает контрольную точку

    Аргументы:
        filename: имя файла
        solver: объект решателя

    Возвращает:
        step: текущий шаг для продолжения
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не найден")

    with h5py.File(filename, 'r') as f:
        step = f.attrs['current_step']

        # Загружаем массивы
        if 'Y' in f:
            solver.Y[:step+1] = f['Y'][:]
        if 'l_1' in f:
            solver.l_1[:step+1] = f['l_1'][:]
        if 'l_2' in f:
            solver.l_2[:step+1] = f['l_2'][:]
        if 'l_3' in f:
            solver.l_3[:step+1] = f['l_3'][:]
        if 'N' in f:
            solver.N[:step+1] = f['N'][:]
        if 'v' in f:
            solver.v[:step+1] = f['v'][:]

        if 'cell_currents' in f:
            if hasattr(solver, 'cell_currents'):
                if solver.cell_currents is not None:
                    solver.cell_currents[:step+1] = f['cell_currents'][:]

    print(f"Контрольная точка загружена из {filename} (шаг {step})")
    return step