"""
Pytest fixtures for cardiac model tests
"""
import pytest
import numpy as np
import scipy.io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.parameters import ModelParameters, EKBParameters, ElectricalParameters, SimulationParameters
from core.state import GlobalState


MAT_FILE_PATH = os.path.join(os.path.dirname(__file__), '120-20mm-Norm(ro081)-SS.mat')


@pytest.fixture(scope='session')
def mat_data():
    """Load MATLAB reference data"""
    mat = scipy.io.loadmat(MAT_FILE_PATH, squeeze_me=True, struct_as_record=False)
    return mat


@pytest.fixture(scope='session')
def mat_params(mat_data):
    """Parameters from MATLAB file"""
    return {
        'IschemiaDeg': int(mat_data['IschemiaDeg']),
        'n': int(mat_data['n']),
        's': int(mat_data['s']),
        'ts': float(mat_data['ts']),
        'dt': float(mat_data['dt']),
        'D': float(mat_data['D']),
        'dx': float(mat_data['dx']),
    }


@pytest.fixture(scope='session')
def default_params():
    """Default ModelParameters"""
    return ModelParameters()


@pytest.fixture(scope='session')
def matlab_params(mat_params):
    """ModelParameters matching MATLAB simulation"""
    ekb_params = EKBParameters()
    elec_params = ElectricalParameters()
    sim_params = SimulationParameters()
    
    sim_params.IschemiaDeg = mat_params['IschemiaDeg']
    sim_params.n = mat_params['n']
    sim_params.s = mat_params['s']
    sim_params.ts = mat_params['ts']
    sim_params.D = mat_params['D']
    
    return ModelParameters(ekb=ekb_params, elec=elec_params, sim=sim_params)


@pytest.fixture
def global_state(default_params):
    """GlobalState with default parameters"""
    return GlobalState(default_params)


@pytest.fixture
def global_state_matlab(matlab_params):
    """GlobalState with MATLAB parameters"""
    return GlobalState(matlab_params)


@pytest.fixture(scope='session')
def y0_reference(mat_data):
    """Initial conditions Y0 from MATLAB"""
    return mat_data['Y0']


@pytest.fixture(scope='session')
def cell_indices():
    """Cell indices for comparison"""
    return [0, 10, 20, 40, 60, 80, 100, 110, 115, 119]


@pytest.fixture(scope='session')
def tolerances():
    """Tolerances for comparison by variable type"""
    return {
        'V': {'abs': 1e-4, 'rel': 1e-6},
        'Ca_i': {'abs': 1e-10, 'rel': 1e-7},
        'Ca_SR': {'abs': 1e-7, 'rel': 2e-7},
        'Na_i': {'abs': 1e-5, 'rel': 1e-6},
        'K_i': {'abs': 1e-4, 'rel': 1e-6},
        'TRPN': {'abs': 1e-9, 'rel': 1e-7},
        'N': {'abs': 1e-8, 'rel': 1e-7},
        'l1': {'abs': 1e-8, 'rel': 1e-7},
        'l2': {'abs': 1e-8, 'rel': 1e-7},
    }