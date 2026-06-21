"""Tests for PLQY.py — absolute PLQY calculations."""
import numpy as np
import pytest

# PLQY depends on thot for experiment management; we test only the pure math functions
from fte_analysis_libraries.General import v_loss, v_sq, qfls, k, T_RT, q


class TestVLoss:
    def test_zero_loss_at_unity_plqy(self):
        assert abs(v_loss(1.0)) < 1e-15

    def test_negative_loss_for_sub_unity_plqy(self):
        assert v_loss(0.5) < 0

    def test_magnitude(self):
        PLQY = 0.1
        expected = k * T_RT / q * np.log(PLQY)
        assert abs(v_loss(PLQY) - expected) < 1e-15


class TestQFLS:
    def test_qfls_equals_vsq_at_unity_plqy(self):
        Eg = 1.6
        result = qfls(Eg, 1.0)
        assert abs(result - v_sq(Eg)) < 1e-10

    def test_qfls_less_than_vsq_for_sub_unity_plqy(self):
        Eg = 1.6
        assert qfls(Eg, 0.5) < v_sq(Eg)

    def test_qfls_plausible_range(self):
        Eg = 1.6
        PLQY = 0.01
        result = qfls(Eg, PLQY)
        # Should still be positive for reasonable PLQY and bandgap
        assert result > 0


class TestVSq:
    def test_increases_with_bandgap(self):
        assert v_sq(1.2) < v_sq(1.6) < v_sq(2.0)

    def test_formula(self):
        Eg = 1.5
        assert abs(v_sq(Eg) - (0.932 * 1.5 - 0.167)) < 1e-12
