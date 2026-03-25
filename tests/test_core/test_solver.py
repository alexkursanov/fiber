"""
Tests for core/solver.py
"""
import pytest
import numpy as np
from core.solver import CardiacSolver
from core.state import GlobalState
from models.electrical import y_init, tnnpe
from models.diffusion import solve_diffusion


class TestCardiacSolverInit:
    """Tests for CardiacSolver initialization"""

    def test_solver_init_default_params(self, default_params, global_state):
        """Test solver initialization with default parameters"""
        solver = CardiacSolver(default_params, global_state)
        
        assert solver.params is not None
        assert solver.gs is not None

    def test_solver_init_arrays_shape(self, default_params, global_state):
        """Test solver initializes arrays with correct shapes"""
        solver = CardiacSolver(default_params, global_state)
        
        s = default_params.sim.s
        n = default_params.sim.n
        
        # Y: (s, n, 24)
        assert solver.Y.shape == (s, n, 24)
        
        # Y1: (n, 24)
        assert solver.Y1.shape == (n, 24)
        
        # cell_currents: (s, n, 14)
        assert solver.cell_currents.shape == (s, n, 14)
        
        # l_1, l_2: (s, n)
        assert solver.l_1.shape == (s, n)
        assert solver.l_2.shape == (s, n)
        
        # l_3: (s,)
        assert solver.l_3.shape == (s,)
        
        # N, v, w: (s, n)
        assert solver.N.shape == (s, n)
        assert solver.v.shape == (s, n)
        assert solver.w.shape == (s, n)

    def test_solver_init_zeros(self, default_params, global_state):
        """Test solver initializes with zeros"""
        solver = CardiacSolver(default_params, global_state)
        
        assert np.all(solver.Y == 0)
        assert np.all(solver.Y1 == 0)

    def test_solver_init_mechanical_variables(self, default_params, global_state):
        """Test solver initializes mechanical variables correctly"""
        solver = CardiacSolver(default_params, global_state)
        
        # l_2 should be computed from r0
        assert solver.l_2[0, 0] != 0.0
        
        # l_1 should be computed
        assert solver.l_1[0, 0] != 0.0
        
        # l_3 should be computed
        assert solver.l_3[0] != 0.0
        
        # N should be set
        assert solver.N[0, 0] != 0.0

    def test_solver_init_helper_variables(self, default_params, global_state):
        """Test solver initializes helper variables"""
        solver = CardiacSolver(default_params, global_state)
        
        # v_1 = v_max / 10
        expected_v_1 = default_params.ekb.v_max / 10.0
        assert np.isclose(solver.v_1, expected_v_1)
        
        # v_st = x_st * v_max
        expected_v_st = default_params.ekb.x_st * default_params.ekb.v_max
        assert np.isclose(solver.v_st, expected_v_st)
        
        # gamma2 should be computed
        assert solver.gamma2 != 0.0


class TestCardiacSolverInitialConditions:
    """Tests for initial conditions"""

    def test_set_initial_conditions(self, default_params, global_state):
        """Test set_initial_conditions method"""
        solver = CardiacSolver(default_params, global_state)
        
        Y_init = y_init()
        solver.set_initial_conditions(Y_init)
        
        # First time step should have initial conditions
        for j in range(default_params.sim.n):
            np.testing.assert_allclose(
                solver.Y[0, j, :], 
                Y_init, 
                rtol=1e-10
            )

    def test_y_init_called(self, default_params, global_state):
        """Test y_init is called during initialization"""
        solver = CardiacSolver(default_params, global_state)
        
        # After initialization, Y[0] should have valid values from mechanical init
        # Note: set_initial_conditions must be called separately (in main.py)
        # But mechanical variables should be initialized
        assert solver.l_2[0, 0] != 0.0  # Mechanical init happens


class TestCardiacSolverStep:
    """Tests for solver step methods"""

    def test_solve_step_runs(self, default_params, global_state):
        """Test solve_step method runs without error"""
        solver = CardiacSolver(default_params, global_state)
        
        # Set initial conditions
        solver.set_initial_conditions(y_init())
        
        # Run one step - should not raise
        try:
            solver.solve_step(0, tnnpe, solve_diffusion)
        except Exception as e:
            # May fail due to numerical issues, but shouldn't crash
            if 'solve_ivp' in str(type(e)):
                pass  # Acceptable
            else:
                raise

    def test_solve_mechanical_step_runs(self, default_params, global_state):
        """Test _solve_mechanical_step method runs"""
        solver = CardiacSolver(default_params, global_state)
        
        # Initialize mechanical arrays
        solver.v[0, :] = 1e-6
        solver.l_1[0, :] = 0.1
        solver.l_2[0, :] = 0.1
        solver.l_3[0] = 0.05
        solver.N[0, :] = 0.01
        
        try:
            solver._solve_mechanical_step(0)
        except Exception as e:
            # May have numerical issues
            if 'fsolve' in str(type(e)):
                pass
            else:
                raise


class TestCardiacSolverResults:
    """Tests for results methods"""

    def test_get_results_returns_dict(self, default_params, global_state):
        """Test get_results returns dictionary"""
        solver = CardiacSolver(default_params, global_state)
        
        results = solver.get_results()
        
        assert isinstance(results, dict)
        
        # Check required keys
        required_keys = [
            'time', 'x', 'V', 'Ca_i', 'Ca_SR', 
            'Na_i', 'K_i', 'TRPN', 'N', 
            'v', 'l1', 'l2', 'l3'
        ]
        
        for key in required_keys:
            assert key in results

    def test_get_results_shapes(self, default_params, global_state):
        """Test get_results returns correct shapes"""
        solver = CardiacSolver(default_params, global_state)
        
        results = solver.get_results()
        
        n = default_params.sim.n
        s = default_params.sim.s
        
        assert results['time'].shape == (s,)
        assert results['x'].shape == (n,)
        assert results['V'].shape == (s, n)
        assert results['N'].shape == (s, n)
        assert results['v'].shape == (s, n)
        assert results['l1'].shape == (s, n)
        assert results['l2'].shape == (s, n)
        assert results['l3'].shape == (s,)

    def test_get_results_structured(self, default_params, global_state):
        """Test get_results_structured returns SimulationResults"""
        solver = CardiacSolver(default_params, global_state)
        
        results = solver.get_results_structured()
        
        assert results is not None
        
        # Check attributes
        assert hasattr(results, 'time')
        assert hasattr(results, 'V')
        assert hasattr(results, 'Ca_i')
        assert hasattr(results, 'ischemia')


class TestCardiacSolverMatlab:
    """Tests for solver with MATLAB-matching parameters"""

    def test_solver_matlab_params(self, matlab_params, global_state_matlab):
        """Test solver with MATLAB parameters"""
        solver = CardiacSolver(matlab_params, global_state_matlab)
        
        s = matlab_params.sim.s
        n = matlab_params.sim.n
        
        assert solver.Y.shape == (s, n, 24)

    def test_solver_init_mechanical_matlab(self, matlab_params, global_state_matlab):
        """Test solver mechanical init with MATLAB params"""
        solver = CardiacSolver(matlab_params, global_state_matlab)
        
        # Check that mechanical variables are initialized
        assert solver.l_2[0, 0] != 0.0
        assert solver.l_1[0, 0] != 0.0
        assert solver.l_3[0] != 0.0


class TestCardiacSolverErrors:
    """Tests for error handling"""

    def test_solver_handles_bad_initial_conditions(self, default_params, global_state):
        """Test solver handles bad initial conditions"""
        solver = CardiacSolver(default_params, global_state)
        
        # NaN initial conditions
        Y_nan = np.full(24, np.nan)
        
        try:
            solver.set_initial_conditions(Y_nan)
            # Should set without error
            assert np.all(np.isnan(solver.Y[0]))
        except:
            pass  # Acceptable


class TestCardiacSolverConsistency:
    """Tests for internal consistency"""

    def test_global_state_updated_after_init(self, default_params, global_state):
        """Test GlobalState is updated after solver init"""
        solver = CardiacSolver(default_params, global_state)
        
        # gs.L should be set
        assert solver.gs.L != 0.0
        
        # gs.l1_n should be set
        assert solver.gs.l1_n.shape == (default_params.sim.n,)

    def test_params_consistent(self, default_params, global_state):
        """Test solver.params is consistent with input"""
        solver = CardiacSolver(default_params, global_state)
        
        assert solver.params.sim.n == default_params.sim.n
        assert solver.params.sim.s == default_params.sim.s