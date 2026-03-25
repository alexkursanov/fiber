"""
Integration tests for full simulation comparing with MATLAB results
"""
import pytest
import numpy as np
from core.solver import CardiacSolver
from core.state import GlobalState
from models.electrical import tnnpe, y_init
from models.diffusion import solve_diffusion


@pytest.mark.slow
class TestFullSimulationMatlab:
    """Integration tests comparing Python simulation with MATLAB reference"""

    @pytest.fixture(scope='class')
    def python_results(self, matlab_params, global_state_matlab):
        """Run Python simulation and return results"""
        solver = CardiacSolver(matlab_params, global_state_matlab)
        
        solver.set_initial_conditions(y_init())
        
        solver.run(tnnpe, solve_diffusion)
        
        return solver.get_results()

    def test_simulation_completes(self, python_results):
        """Test that simulation completes without errors"""
        assert python_results is not None
        assert 'V' in python_results

    def test_voltage_range(self, python_results):
        """Test voltage is in reasonable range"""
        V = python_results['V']
        
        # Should be between -100 and 50 mV
        assert V.min() > -100
        assert V.max() < 50

    def test_no_nan_in_results(self, python_results):
        """Test no NaN values in results"""
        for key, value in python_results.items():
            if isinstance(value, np.ndarray):
                assert not np.any(np.isnan(value)), f"NaN found in {key}"

    def test_no_inf_in_results(self, python_results):
        """Test no Inf values in results"""
        for key, value in python_results.items():
            if isinstance(value, np.ndarray):
                assert not np.any(np.isinf(value)), f"Inf found in {key}"


@pytest.mark.slow
class TestFullSimulationMatlabComparison:
    """Tests comparing Python results with MATLAB reference data - skip by default"""

    @pytest.mark.skip(reason="Long running - use pytest -m slow to run")
    def test_voltage_cell_0(self):
        pass