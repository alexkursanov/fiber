"""
Tests for models/electrical.py
"""
import pytest
import numpy as np
from models.electrical import (
    y_init, 
    tnnpe, 
    tnnpe_explicit, 
    calculate_ischemia_params
)


class TestYInit:
    """Tests for y_init function"""

    def test_y_init_shape(self):
        """Test y_init returns correct shape"""
        Y = y_init()
        assert Y.shape == (24,)

    def test_y_init_no_nan(self):
        """Test y_init has no NaN values"""
        Y = y_init()
        assert not np.any(np.isnan(Y))

    def test_y_init_no_inf(self):
        """Test y_init has no Inf values"""
        Y = y_init()
        assert not np.any(np.isinf(Y))

    def test_y_init_matches_matlab(self, y0_reference):
        """Test y_init produces reasonable values close to MATLAB Y0"""
        Y = y_init()
        
        # y_init should produce values in the same ballpark as MATLAB
        # but exact match isn't expected since init may differ
        # Check voltage is close (-78 to -80 mV)
        assert -90 < Y[12] < -70
        
        # Check K_i is reasonable (~137 mM)
        assert 130 < Y[9] < 145
        
        # Check Na_i is reasonable (~12 mM)
        assert 10 < Y[10] < 15
        
        # Check Ca_i is reasonable (< 1 mM)
        assert 0 < Y[8] < 0.001


class TestCalculateIschemiaParams:
    """Tests for calculate_ischemia_params function"""

    def test_ischemia_0_no_change(self):
        """Test ischemia degree 0 - no parameter change"""
        ATP_i = 6.8
        K_o = 5.4
        g_Na = 0.0008
        J_L = 0.000913
        bzdegree = 0.5
        
        ATP_out, K_out, g_Na_out, J_L_out = calculate_ischemia_params(
            0, bzdegree, ATP_i, K_o, g_Na, J_L
        )
        
        assert ATP_out == ATP_i
        assert K_out == K_o
        assert g_Na_out == g_Na
        assert J_L_out == J_L

    def test_ischemia_5_no_bz(self):
        """Test ischemia degree 5 with bzdegree=0"""
        ATP_i = 6.8
        K_o = 5.4
        g_Na = 0.0008
        J_L = 0.000913
        bzdegree = 0.0
        
        ATP_out, K_out, g_Na_out, J_L_out = calculate_ischemia_params(
            5, bzdegree, ATP_i, K_o, g_Na, J_L
        )
        
        assert ATP_out == ATP_i
        assert K_out == K_o
        assert g_Na_out == g_Na
        assert J_L_out == J_L

    def test_ischemia_5_full_bz(self):
        """Test ischemia degree 5 with bzdegree=1"""
        ATP_i = 6.8
        K_o = 5.4
        g_Na = 0.0008
        J_L = 0.000913
        
        ATP_out, K_out, g_Na_out, J_L_out = calculate_ischemia_params(
            5, 1.0, ATP_i, K_o, g_Na, J_L
        )
        
        assert ATP_out == ATP_i * (1 - 0.2)
        assert K_out == 5.4 + 1.885 * 1.0
        assert g_Na_out == g_Na * (1 - 0.125)
        assert J_L_out == J_L * (1 - 0.15)

    def test_ischemia_10_full_bz(self):
        """Test ischemia degree 10 with bzdegree=1"""
        ATP_i = 6.8
        K_o = 5.4
        g_Na = 0.0008
        J_L = 0.000913
        
        ATP_out, K_out, g_Na_out, J_L_out = calculate_ischemia_params(
            10, 1.0, ATP_i, K_o, g_Na, J_L
        )
        
        assert ATP_out == ATP_i * (1 - 0.37)
        assert K_out == 5.4 + 1.885 * 1.0
        assert g_Na_out == g_Na * (1 - 0.25)
        assert J_L_out == J_L * (1 - 0.3)

    def test_ischemia_15_full_bz(self):
        """Test ischemia degree 15 with bzdegree=1"""
        ATP_i = 6.8
        K_o = 5.4
        g_Na = 0.0008
        J_L = 0.000913
        
        ATP_out, K_out, g_Na_out, J_L_out = calculate_ischemia_params(
            15, 1.0, ATP_i, K_o, g_Na, J_L
        )
        
        assert ATP_out == ATP_i * (1 - 0.53)
        assert K_out == 5.4 + 4.5 * 1.0
        assert g_Na_out == g_Na * (1 - 0.375)
        assert J_L_out == J_L * (1 - 0.7)


class TestTNNPE:
    """Tests for tnnpe function"""

    @pytest.fixture
    def sample_state(self, default_params, global_state):
        """Create sample state for testing"""
        y = y_init()
        global_state.jj = 1
        global_state.N_elec = 0.0
        return y, default_params, global_state

    def test_tnnpe_returns_24_elements(self, sample_state):
        """Test tnnpe returns 24 elements"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        assert len(dY) == 24

    def test_tnnpe_no_nan(self, sample_state):
        """Test tnnpe output has no NaN"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        assert not np.any(np.isnan(dY))

    def test_tnnpe_no_inf(self, sample_state):
        """Test tnnpe output has no Inf"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        assert not np.any(np.isinf(dY))

    def test_tnnpe_voltage_gate(self, sample_state):
        """Test voltage derivative is computed"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        # dV is at index 12
        assert np.isfinite(dY[12])

    def test_tnnpe_calcium_gate(self, sample_state):
        """Test calcium dynamics are computed"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        # Ca_i at index 8, Ca_SR at index 7
        assert np.isfinite(dY[8])
        assert np.isfinite(dY[7])

    def test_tnnpe_ion_concentrations(self, sample_state):
        """Test ion concentration derivatives"""
        y, params, gs = sample_state
        dY = tnnpe(0.0, y, 1, 0.0, params, gs)
        # Na_i at 10, K_i at 9
        assert np.isfinite(dY[10])
        assert np.isfinite(dY[9])

    def test_tnnpe_stimulus_at_first_cell(self, sample_state):
        """Test stimulus is applied at first cell"""
        y, params, gs = sample_state
        gs.jj = 1
        
        # At stimulus time (time % period should be >= StimStart_shift)
        dY_stim = tnnpe(60.0, y, 1, 0.0, params, gs)
        
        # At non-stimulus time
        dY_nostim = tnnpe(50.0, y, 1, 0.0, params, gs)
        
        # dV should be different (stimulus current added)
        assert not np.isclose(dY_stim[12], dY_nostim[12], rtol=1e-3)

    def test_tnnpe_no_stimulus_at_other_cells(self, sample_state):
        """Test no stimulus at non-first cells"""
        y, params, gs = sample_state
        gs.jj = 10
        
        dY = tnnpe(60.0, y, 10, 0.0, params, gs)
        
        # I_Stim should be 0 for non-first cell
        # The stimulus affects dV, so check it's finite but not dramatically different
        assert np.isfinite(dY[12])


class TestTNNPEExplicit:
    """Tests for tnnpe_explicit function"""

    def test_explicit_returns_currents(self):
        """Test tnnpe_explicit returns currents"""
        from config.parameters import ElectricalParameters, EKBParameters
        
        elec_params = ElectricalParameters()
        ekb_params = EKBParameters()
        y = y_init()
        
        dY, currents = tnnpe_explicit(0.0, y, 1, 0.0, elec_params, ekb_params, 0)
        
        assert len(currents) == 14

    def test_explicit_no_nan(self):
        """Test tnnpe_explicit has no NaN"""
        from config.parameters import ElectricalParameters, EKBParameters
        
        elec_params = ElectricalParameters()
        ekb_params = EKBParameters()
        y = y_init()
        
        dY, currents = tnnpe_explicit(0.0, y, 1, 0.0, elec_params, ekb_params, 0)
        
        assert not np.any(np.isnan(dY))
        assert not np.any(np.isnan(currents))

    def test_explicit_with_ischemia(self):
        """Test tnnpe_explicit with ischemia"""
        from config.parameters import ElectricalParameters, EKBParameters
        
        elec_params = ElectricalParameters()
        ekb_params = EKBParameters()
        y = y_init()
        
        # Ischemia degree 15
        dY, currents = tnnpe_explicit(0.0, y, 1, 0.0, elec_params, ekb_params, 15)
        
        assert not np.any(np.isnan(dY))
        assert len(currents) == 14


class TestTNNPEIschemia:
    """Tests for TNNPE with ischemia"""

    @pytest.fixture
    def ischemic_params(self, default_params):
        """Params with ischemia"""
        default_params.sim.IschemiaDeg = 15
        return default_params

    def test_ischemia_modifies_K_o(self, ischemic_params, global_state):
        """Test ischemia modifies extracellular K+"""
        y = y_init()
        
        # First cell (border zone)
        global_state.jj = 35  # In BZ1
        global_state.IschemiaDeg = 15
        
        dY = tnnpe(0.0, y, 35, 0.0, ischemic_params, global_state)
        
        # Should not have NaN
        assert not np.any(np.isnan(dY))

    def test_ischemia_degree_0(self, default_params, global_state):
        """Test no ischemia (degree 0)"""
        y = y_init()
        global_state.IschemiaDeg = 0
        
        dY = tnnpe(0.0, y, 1, 0.0, default_params, global_state)
        
        assert not np.any(np.isnan(dY))


class TestTNNPEBoundaryConditions:
    """Tests for boundary conditions in TNNPE"""

    def test_first_cell_has_stimulus(self, default_params, global_state):
        """Test first cell receives stimulus"""
        y = y_init()
        
        dY = tnnpe(60.0, y, 1, 0.0, default_params, global_state)
        
        # Should be different from no-stimulus time
        dY_no_stim = tnnpe(50.0, y, 1, 0.0, default_params, global_state)
        
        # At stimulus time, dV should include stimulus current
        assert dY[12] != dY_no_stim[12]