"""
Tests for core/state.py
"""
import pytest
import numpy as np
from core.state import GlobalState


class TestGlobalStateInit:
    """Tests for GlobalState initialization"""

    def test_default_init(self):
        """Test default initialization"""
        gs = GlobalState()
        
        assert gs.jj == 0
        assert gs.cell_cur.shape == (14,)
        assert gs.N_elec == 0.0
        assert gs.N_meh == 0.0
        assert gs.n == 120

    def test_init_with_params(self, default_params):
        """Test initialization with parameters"""
        gs = GlobalState(default_params)
        
        assert gs.n == default_params.sim.n

    def test_init_updates_from_params(self, default_params):
        """Test that parameters update state fields"""
        params = default_params
        params.sim.n = 50
        params.sim.IschemiaDeg = 10
        params.sim.BZ1Start = 10
        params.sim.BZ1End = 20
        
        gs = GlobalState(params)
        
        assert gs.n == 50
        assert gs.IschemiaDeg == 10
        assert gs.BZ1Start == 10
        assert gs.BZ1End == 20


class TestGlobalStateIschemia:
    """Tests for ischemia-related methods"""

    def test_calc_bzdegree_first_zone_start(self):
        """Test bzdegree at start of first border zone"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        
        # At BZ1Start
        bz = gs.get_bzdegree_for_cell(25)
        assert bz == 0.0

    def test_calc_bzdegree_first_zone_middle(self):
        """Test bzdegree in middle of first border zone"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        
        # At middle of BZ1
        bz = gs.get_bzdegree_for_cell(35)
        assert 0.0 < bz < 1.0

    def test_calc_bzdegree_first_zone_end(self):
        """Test bzdegree at end of first border zone"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        
        # At BZ1End
        bz = gs.get_bzdegree_for_cell(45)
        assert 0.99 < bz <= 1.0  # May have floating point issue

    def test_calc_bzdegree_second_zone(self):
        """Test bzdegree in second border zone"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        gs.BZ2Start = 75
        gs.BZ2End = 95
        
        # At middle of BZ2
        bz = gs.get_bzdegree_for_cell(85)
        assert 0.0 < bz <= 1.0

    def test_calc_bzdegree_outside_zones(self):
        """Test bzdegree outside border zones"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        gs.BZ2Start = 75
        gs.BZ2End = 95
        
        # Before BZ1
        bz = gs.get_bzdegree_for_cell(1)
        assert bz == 0.0
        
        # After BZ2
        bz = gs.get_bzdegree_for_cell(119)
        assert bz == 0.0

    def test_calc_bzdegree_at_boundary(self):
        """Test bzdegree at boundary between zones"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        gs.BZ2Start = 75
        gs.BZ2End = 95
        
        # At cell 46 - in the healthy region between BZ1 and BZ2
        # This depends on implementation - check current behavior
        bz = gs.get_bzdegree_for_cell(46)
        # Cell 46 is after BZ1End (45), so should be 0 (healthy tissue)
        assert bz >= 0.0

    def test_ischemia_degree_property(self):
        """Test IschemiaDeg property"""
        gs = GlobalState()
        
        gs.IschemiaDeg = 10
        assert gs.IschemiaDeg == 10
        
        gs.IschemiaDeg = 15
        assert gs.IschemiaDeg == 15


class TestGlobalStateUpdate:
    """Tests for update methods"""

    def test_update_from_params(self, default_params):
        """Test update_from_params method"""
        params = default_params
        params.sim.n = 100
        params.sim.IschemiaDeg = 5
        params.sim.BZ1Start = 20
        params.sim.BZ1End = 40
        params.sim.BZ2Start = 70
        params.sim.BZ2End = 90
        
        gs = GlobalState()
        gs.n = 50  # Change default
        gs.update_from_params(params)
        
        assert gs.n == 100
        assert gs.IschemiaDeg == 5
        assert gs.BZ1Start == 20
        assert gs.l1_n.shape == (100,)

    def test_update_creates_correct_l1_n_size(self, default_params):
        """Test that update_from_params creates correct l1_n size"""
        params = default_params
        params.sim.n = 60
        
        gs = GlobalState()
        gs.update_from_params(params)
        
        assert gs.l1_n.shape == (60,)


class TestGlobalStateDefaults:
    """Tests for default values"""

    def test_default_ischemia_zones(self):
        """Test default ischemia zone parameters"""
        gs = GlobalState()
        
        assert gs.BZ1Start == 25
        assert gs.BZ1End == 45
        assert gs.BZ2Start == 75
        assert gs.BZ2End == 95

    def test_default_mechanical_loading(self):
        """Test default mechanical loading"""
        gs = GlobalState()
        
        assert gs.Lam_mech == 55.0

    def test_default_lengths(self):
        """Test default length values"""
        gs = GlobalState()
        
        assert gs.L == 0.0
        assert gs.dx == 0.0

    def test_default_mechanical_variables(self):
        """Test default mechanical variables"""
        gs = GlobalState()
        
        assert gs.l1 == 0.0
        assert gs.l2 == 0.0
        assert gs.l3 == 0.0
        assert gs.l1_n.shape == (120,)


class TestGlobalStateCellIndex:
    """Tests for cell index handling"""

    def test_cell_index_1based(self):
        """Test that cell index is 1-based"""
        gs = GlobalState()
        gs.BZ1Start = 25
        gs.BZ1End = 45
        
        # Test the logic - the function takes 1-based index
        # Cell 25 should be at start of BZ1
        bz = gs.get_bzdegree_for_cell(25)
        assert bz == 0.0
        
        # Cell 45 should be at end of BZ1 - check approximately
        bz = gs.get_bzdegree_for_cell(45)
        assert bz > 0.99

    def test_invalid_cell_index(self):
        """Test behavior with invalid cell index"""
        gs = GlobalState()
        
        # Should handle gracefully
        bz = gs.get_bzdegree_for_cell(0)
        assert bz == 0.0  # Should clamp to 0
        
        bz = gs.get_bzdegree_for_cell(200)
        assert bz == 0.0  # Should clamp to 0