"""Tests for Electrochemistry.py — EIS utilities."""
import numpy as np
import pytest

from fte_analysis_libraries.Electrochemistry import EIS_predict, Z_in_4th_quadrant


class TestZIn4thQuadrant:
    def test_all_in_4th_quadrant(self):
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        Z = np.array([10.0 - 5.0j, 8.0 - 3.0j, 6.0 - 1.0j, 5.0 - 0.5j])
        result = Z_in_4th_quadrant(f, Z)
        # all imaginary parts negative → all in 4th quadrant
        assert len(result[0]) == 4

    def test_filters_out_positive_imag(self):
        f = np.array([1.0, 10.0, 100.0])
        # Mix of 4th (neg imag) and 1st (pos imag)
        Z = np.array([10.0 - 5.0j, 8.0 + 2.0j, 6.0 - 1.0j])
        result = Z_in_4th_quadrant(f, Z)
        f_out, Z_out = result
        assert len(f_out) == 2
        assert all(np.imag(Z_out) <= 0)


class TestEISPredict:
    def test_series_rc_prediction(self):
        # Series RC: Z = R + 1/(j*omega*C)
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        R = 50.0
        C = 1e-6
        circuit_str = 'R0-C0'
        params = [R, C]
        Z = EIS_predict(f, circuit_str, params)
        assert len(Z) == len(f)
        # At each frequency: Re(Z) should be close to R
        np.testing.assert_allclose(np.real(Z), R, atol=0.01)

    def test_prediction_returns_complex(self):
        f = np.array([100.0, 1000.0])
        circuit_str = 'R0-C0'
        params = [100.0, 1e-6]
        Z = EIS_predict(f, circuit_str, params)
        assert np.iscomplexobj(Z)
