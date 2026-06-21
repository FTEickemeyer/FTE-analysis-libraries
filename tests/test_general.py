"""Tests for General.py utility functions."""
import numpy as np
import pandas as pd
import pytest

from fte_analysis_libraries.General import (
    T_RT,
    df_interpolate,
    diff_coeff,
    findind,
    idx_range,
    int_arr,
    interpolated_array,
    is_even,
    is_odd,
    k,
    linfit,
    mobility,
    q,
    qfls,
    round_sig,
    str_round_sig,
    v_loss,
    v_sq,
)


class TestFindind:
    def test_exact_value(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert findind(arr, 2.0) == 2

    def test_first_value_gte(self):
        # findind returns first index where arr[i] >= value
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert findind(arr, 1.4) == 2

    def test_below_min_clamps(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert findind(arr, 0.0) == 0

    def test_above_max_clamps(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert findind(arr, 10.0) == 2

    def test_descending_array(self):
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert findind(arr, 3.0) == 2


class TestLinfit:
    def test_perfect_line(self):
        x = np.linspace(0, 10, 100)
        y = 2.5 * x + 1.3
        m, b = linfit(x, y)
        assert abs(m - 2.5) < 1e-8
        assert abs(b - 1.3) < 1e-8

    def test_with_range(self):
        x = np.linspace(0, 10, 101)
        y = 3.0 * x - 2.0
        m, b = linfit(x, y, von=2.0, bis=8.0)
        assert abs(m - 3.0) < 1e-6
        assert abs(b - (-2.0)) < 1e-6

    def test_slope_zero(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])
        m, b = linfit(x, y)
        assert abs(m) < 1e-10
        assert abs(b - 5.0) < 1e-10


class TestRoundSig:
    def test_two_sig_figs(self):
        assert round_sig(3.14159, 2) == 3.1

    def test_zero(self):
        assert round_sig(0, 2) == 0

    def test_large_number(self):
        assert round_sig(12345.6, 3) == 12300.0

    def test_small_number(self):
        result = round_sig(0.001234, 2)
        assert abs(result - 0.0012) < 1e-12


class TestStrRoundSig:
    def test_returns_string(self):
        result = str_round_sig(3.14159, 3)
        assert isinstance(result, str)

    def test_correct_value(self):
        result = str_round_sig(3.14159, 3)
        assert result == '3.14'

    def test_zero(self):
        result = str_round_sig(0, 2)
        assert result == '0.0'


class TestIntArr:
    def test_interpolation(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        new_x = np.array([0.5, 1.5, 2.5])
        result = int_arr(x, y, new_x)
        assert len(result) == 3

    def test_identity_on_original_points(self):
        x = np.linspace(0, 10, 50)
        y = 2 * x
        result = int_arr(x, y, x, kind='linear')
        np.testing.assert_allclose(result, y, atol=1e-10)


class TestInterpolatedArray:
    def test_linear_interpolation(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        result = interpolated_array(x, y, np.array([0.5, 1.5]))
        np.testing.assert_allclose(result, [1.0, 3.0], atol=1e-10)


class TestDfInterpolate:
    def test_basic_interpolation(self):
        idx = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        df = pd.DataFrame({'a': [0.0, 1.0, 4.0, 9.0, 16.0]}, index=idx)
        new_idx = np.array([0.5, 1.5, 2.5])
        result = df_interpolate(df, new_idx)
        assert len(result) == 3
        assert list(result.index) == [0.5, 1.5, 2.5]

    def test_clamps_to_bounds(self):
        idx = np.array([1.0, 2.0, 3.0])
        df = pd.DataFrame({'a': [10.0, 20.0, 30.0]}, index=idx)
        new_idx = np.array([0.0, 1.5, 4.0])
        result = df_interpolate(df, new_idx)
        # out-of-bounds values are excluded
        assert 1.5 in result.index
        assert 0.0 not in result.index
        assert 4.0 not in result.index


class TestIsEvenIsOdd:
    def test_even(self):
        from fte_analysis_libraries.General import is_even, is_odd
        assert is_even(4) is True
        assert is_even(3) is False

    def test_odd(self):
        from fte_analysis_libraries.General import is_even, is_odd
        assert is_odd(3) is True
        assert is_odd(4) is False


class TestIdxRange:
    def test_full_range(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = idx_range(arr)
        assert list(r) == list(range(len(arr)))

    def test_partial_range(self):
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        r = idx_range(arr, left=1.0, right=3.0)
        assert 1 in r
        assert 3 in r
        assert 4 not in r


class TestPhysicsHelpers:
    def test_v_sq(self):
        Eg = 1.6  # eV
        result = v_sq(Eg)
        assert abs(result - (0.932 * 1.6 - 0.167)) < 1e-10

    def test_v_loss_plqy_1(self):
        # PLQY=1 → ln(1)=0 → zero voltage loss
        result = v_loss(1.0)
        assert abs(result) < 1e-15

    def test_v_loss_plqy_small(self):
        PLQY = 0.01
        expected = k * T_RT / q * np.log(PLQY)
        assert abs(v_loss(PLQY) - expected) < 1e-15

    def test_qfls(self):
        Eg = 1.6
        PLQY = 1.0
        result = qfls(Eg, PLQY)
        assert abs(result - v_sq(Eg)) < 1e-10

    def test_diff_coeff_mobility_roundtrip(self):
        mu = 10.0  # cm2/(Vs)
        D = diff_coeff(mu)
        mu_back = mobility(D)
        assert abs(mu_back - mu) < 1e-10
