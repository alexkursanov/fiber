"""
Tests for config/parameters.py
"""
import pytest
import numpy as np
from config.parameters import ModelParameters, EKBParameters, ElectricalParameters, SimulationParameters


class TestEKBParameters:
    """Tests for EKBParameters dataclass"""

    def test_default_values(self):
        """Test default parameter values"""
        params = EKBParameters()
        
        # Mechanical parameters
        assert params.alpha_1 == 21.0
        assert params.beta_1 == 0.94
        assert params.alpha_2 == 14.6
        assert params.beta_2 == 0.0018
        assert params.alpha_3 == 33.79
        assert params.beta_3 == 0.0084
        
        # Kinetic parameters
        assert params.q_1 == 0.0173
        assert params.q_4 == 0.015
        assert params.v_max == 0.0055
        assert params.a == 0.25
        
        # Geometry
        assert params.r0 == 0.081
        assert params.llambda == 55.0

    def test_apply_nondim(self):
        """Test nondimensionalization"""
        params = EKBParameters()
        original_alpha_1 = params.alpha_1
        
        params.apply_nondim()
        
        # alpha values should be multiplied by L_0 (1.67)
        assert params.alpha_1 != original_alpha_1


class TestElectricalParameters:
    """Tests for ElectricalParameters dataclass"""

    def test_default_values(self):
        """Test default parameter values"""
        params = ElectricalParameters()
        
        # Cell geometry
        assert params.V_myo == 25850.0
        assert params.V_SR_uL == 2.098e-6
        
        # Ion concentrations
        assert params.Ca_o == 1.2
        assert params.K_o == 5.4
        assert params.Na_o == 140.0
        assert params.ATP_i == 6.8
        
        # Stimulation
        assert params.stim_period == 1000.0
        assert params.stim_amplitude == -0.0006

    def test_computed_properties(self):
        """Test computed properties"""
        params = ElectricalParameters()
        
        # K_o_norm
        assert params.K_o_norm == 5.4
        
        # t_R
        assert params.t_R == 1.17 * params.t_L
        
        # alpha_m, beta_m
        assert params.alpha_m == params.phi_L / params.t_L
        assert params.beta_m == params.phi_R / params.t_R
        
        # g_Na_endo
        assert params.g_Na_endo == 1.33 * params.g_Na
        
        # sigma
        assert params.sigma == (np.exp(params.Na_o / 67.3) - 1.0) / 7.0


class TestSimulationParameters:
    """Tests for SimulationParameters dataclass"""

    def test_default_values(self):
        """Test default parameter values"""
        params = SimulationParameters()
        
        assert params.t0 == 0.0
        assert params.ts == 1000.0
        assert params.s == 1000
        assert params.n == 80
        assert params.D == 150.0
        assert params.IschemiaDeg == 15

    def test_derived_parameters(self):
        """Test computed derived parameters"""
        params = SimulationParameters()
        
        # dt
        expected_dt = (params.ts - params.t0) / (params.s - 1)
        assert np.isclose(params.dt, expected_dt)
        
        # dx
        expected_dx = (params.xn - params.x0) / (params.n - 1)
        assert np.isclose(params.dx, expected_dx)
        
        # t and x arrays
        assert len(params.t) == params.s
        assert len(params.x) == params.n

    def test_update_triggers_recompute(self):
        """Test that updating parameters triggers recompute"""
        params = SimulationParameters()
        
        original_dt = params.dt
        params.ts = 2000.0
        
        assert params.dt != original_dt
        assert params.dt == (2000.0 - 0.0) / (params.s - 1)

    def test_ischemia_zones(self):
        """Test ischemia zone parameters"""
        params = SimulationParameters()
        
        assert params.BZ1Start == 25
        assert params.BZ1End == 45
        assert params.BZ2Start == 75
        assert params.BZ2End == 95


class TestModelParameters:
    """Tests for ModelParameters dataclass"""

    def test_default_creation(self):
        """Test default model parameters creation"""
        params = ModelParameters()
        
        assert isinstance(params.ekb, EKBParameters)
        assert isinstance(params.elec, ElectricalParameters)
        assert isinstance(params.sim, SimulationParameters)

    def test_custom_params(self):
        """Test creating with custom parameters"""
        ekb = EKBParameters(v_max=0.01)
        elec = ElectricalParameters(ATP_i=5.0)
        sim = SimulationParameters(n=100, s=500)
        
        # Should not raise
        params = ModelParameters(ekb=ekb, elec=elec, sim=sim)
        
        # Check non-mechanical parameters
        assert params.elec.ATP_i == 5.0
        assert params.sim.n == 100
        assert params.sim.s == 500

    def test_post_init_nondim(self):
        """Test that apply_nondim is called in post_init"""
        params = ModelParameters()
        
        # Check that nondim was applied (alpha values changed)
        assert params.ekb.alpha_1 != 21.0  # Original value is 21.0


class TestSimulationParametersMatlab:
    """Tests for SimulationParameters matching MATLAB"""

    @pytest.fixture
    def matlab_sim_params(self):
        """SimulationParameters matching MATLAB file"""
        params = SimulationParameters()
        params.IschemiaDeg = 0
        params.n = 120
        params.s = 2000
        params.ts = 1000.0
        params.D = 300.0
        return params

    def test_dt_calculation(self, matlab_sim_params):
        """Test dt matches MATLAB"""
        expected_dt = 0.5002501250625313
        assert np.isclose(matlab_sim_params.dt, expected_dt, rtol=1e-6)

    def test_dx_calculation(self, matlab_sim_params):
        """Test dx matches MATLAB"""
        expected_dx = 0.008403361344537815
        assert np.isclose(matlab_sim_params.dx, expected_dx, rtol=1e-6)

    def test_array_lengths(self, matlab_sim_params):
        """Test array lengths"""
        assert len(matlab_sim_params.t) == 2000
        assert len(matlab_sim_params.x) == 120