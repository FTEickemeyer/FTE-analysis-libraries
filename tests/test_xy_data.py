"""Tests for XYdata.py — XYData and MXYData classes."""
import os
import tempfile

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fte_analysis_libraries.XYdata import MXYData, XYData

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make(n=50, slope=2.0, offset=0.0, x0=0.0, x1=10.0):
    x = np.linspace(x0, x1, n)
    y = slope * x + offset
    return XYData(x, y)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestXYDataConstruction:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        d = XYData(x, y)
        np.testing.assert_array_equal(d.x, x)
        np.testing.assert_array_equal(d.y, y)

    def test_with_dict_quants_units(self):
        x = np.linspace(0, 1, 10)
        y = x ** 2
        d = XYData(x, y, quants={'x': 'Wavelength', 'y': 'Intensity'},
                   units={'x': 'nm', 'y': 'counts'})
        assert d.qx == 'Wavelength'
        assert d.ux == 'nm'
        assert d.qy == 'Intensity'
        assert d.uy == 'counts'

    def test_with_list_quants_units(self):
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        d = XYData(x, y, quants=['xq', 'yq'], units=['xu', 'yu'])
        assert d.qx == 'xq'
        assert d.qy == 'yq'
        assert d.ux == 'xu'
        assert d.uy == 'yu'

    def test_with_name(self):
        d = XYData(np.array([0.0, 1.0]), np.array([0.0, 1.0]), name='test')
        assert d.name == 'test'

    def test_no_check_data(self):
        d = XYData(np.array([1.0, 2.0]), np.array([1.0, 2.0]), check_data=False)
        assert len(d.x) == 2

    def test_data_check_bad_order(self, capsys):
        # descending x should trigger warning
        XYData(np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0]))
        out = capsys.readouterr().out
        assert 'ascending' in out.lower() or 'order' in out.lower() or 'attention' in out.lower()

    def test_data_check_nan(self, capsys):
        XYData(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
        out = capsys.readouterr().out
        assert 'nan' in out.lower() or 'attention' in out.lower()

    def test_generate_empty(self):
        d = XYData.generate_empty()
        assert len(d.x) == 0
        assert len(d.y) == 0


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------
class TestXYDataCopy:
    def test_independence(self):
        d = _make()
        c = d.copy()
        c.x[0] = 999.0
        assert d.x[0] != 999.0

    def test_name_copied(self):
        d = XYData(np.array([1.0, 2.0]), np.array([1.0, 2.0]), name='my')
        assert d.copy().name == 'my'


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
class TestXYDataMetadata:
    def test_quants_returns_dict(self):
        d = XYData(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                   quants={'x': 'Wavelength', 'y': 'Intensity'},
                   units={'x': 'nm', 'y': 'cts'})
        q = d.quants()
        assert q['x'] == 'Wavelength'
        assert q['y'] == 'Intensity'

    def test_units_returns_dict(self):
        d = XYData(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                   units={'x': 'nm', 'y': 'cts'})
        u = d.units()
        assert u['x'] == 'nm'
        assert u['y'] == 'cts'

    def test_qy_uy_setter(self):
        d = _make()
        d.qy_uy('Intensity', 'cps')
        assert d.qy == 'Intensity'
        assert d.uy == 'cps'


# ---------------------------------------------------------------------------
# from_df / to_df
# ---------------------------------------------------------------------------
class TestXYDataFromDf:
    def test_from_df_no_quants(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': [4.0, 5.0, 6.0]})
        d = XYData.from_df(df, y_col=0, take_quants_and_units_from_df=False)
        assert len(d.x) == 3

    def test_from_df_second_col(self):
        df = pd.DataFrame({'A': [1.0, 2.0], 'B': [10.0, 20.0]})
        d = XYData.from_df(df, y_col=1, take_quants_and_units_from_df=False)
        np.testing.assert_array_equal(d.y, np.array([10.0, 20.0]))

    def test_from_df_with_quants(self):
        df = pd.DataFrame({'y (cts)': [1.0, 2.0, 3.0]})
        df.index = pd.Index([0.0, 1.0, 2.0], name='x (nm)')
        d = XYData.from_df(df, y_col=0, take_quants_and_units_from_df=True)
        assert d.qx == 'x'
        assert d.ux == 'nm'
        assert d.qy == 'y'
        assert d.uy == 'cts'

    def test_to_df_roundtrip(self):
        fp = os.path.join(FIXTURES, 'xy_data.csv')
        d = XYData.load(fp)
        df = d.to_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
class TestXYDataLoad:
    def test_load_csv(self):
        fp = os.path.join(FIXTURES, 'xy_data.csv')
        d = XYData.load(fp)
        assert len(d.x) == 6
        np.testing.assert_allclose(d.y, 2 * d.x, atol=1e-10)

    def test_load_directory(self):
        # When loading a directory, takes the first file
        dir_path = FIXTURES
        # Just ensure it doesn't crash
        d = XYData.load(dir_path)
        assert d is not None

    def test_load_with_take_quants(self):
        fp = os.path.join(FIXTURES, 'xy_data.csv')
        d = XYData.load(fp, take_quants_and_units_from_file=True)
        assert d.qx == 'x'
        assert d.qy == 'y'

    def test_load_with_name_override(self):
        fp = os.path.join(FIXTURES, 'xy_data.csv')
        d = XYData.load(fp, name='my_custom_name')
        assert d.name == 'my_custom_name'

    def test_load_invalid_path_returns_none(self):
        d = XYData.load('/nonexistent/path/file.csv')
        assert d is None


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
class TestXYDataSave:
    def test_save_no_check(self):
        d = XYData(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]),
                   units={'x': 'nm', 'y': 'cts'})
        with tempfile.TemporaryDirectory() as tmpdir:
            d.save(tmpdir, 'test_out.csv', check_existing=False)
            out_path = os.path.join(tmpdir, 'test_out.csv')
            assert os.path.exists(out_path)
            loaded = pd.read_csv(out_path)
            assert len(loaded) == 3


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------
class TestXYDataArithmetic:
    def _pair(self):
        x = np.linspace(0, 5, 50)
        a = XYData(x.copy(), np.ones(50) * 3.0)
        b = XYData(x.copy(), np.ones(50) * 2.0)
        return a, b

    def test_mul_scalar(self):
        d = _make()
        r = d * 2
        np.testing.assert_allclose(r.y, d.y * 2, atol=1e-10)

    def test_mul_same_type(self):
        a, b = self._pair()
        r = a * b
        np.testing.assert_allclose(r.y, 6.0, atol=0.01)

    def test_add_scalar(self):
        d = _make()
        r = d + 10.0
        np.testing.assert_allclose(r.y, d.y + 10.0, atol=1e-10)

    def test_add_same_type(self):
        a, b = self._pair()
        r = a + b
        np.testing.assert_allclose(r.y, 5.0, atol=0.01)

    def test_sub_scalar(self):
        d = _make()
        r = d - 1.0
        np.testing.assert_allclose(r.y, d.y - 1.0, atol=1e-10)

    def test_sub_same_type(self):
        a, b = self._pair()
        r = a - b
        np.testing.assert_allclose(r.y, 1.0, atol=0.01)

    def test_div_scalar(self):
        d = _make()
        r = d / 2.0
        np.testing.assert_allclose(r.y, d.y / 2.0, atol=1e-10)

    def test_div_same_type(self):
        a, b = self._pair()
        r = a / b
        np.testing.assert_allclose(r.y, 1.5, atol=0.01)


# ---------------------------------------------------------------------------
# y_of / x_of
# ---------------------------------------------------------------------------
class TestXYDataYOf:
    def test_y_of_exact(self):
        d = _make(slope=2.0, offset=0.0)
        # At x=5: y=10
        y = d.y_of(5.0)
        assert abs(y - 10.0) < 0.5

    def test_y_of_interpolated(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        d = XYData(x, y)
        # at x=1.5 interpolated = 3.0
        y_val = d.y_of(1.5, interpolate=True)
        assert abs(y_val - 3.0) < 0.1

    def test_y_of_below_min(self, capsys):
        d = _make()
        y = d.y_of(-1.0)
        assert isinstance(y, float)

    def test_y_of_above_max(self, capsys):
        d = _make()
        y = d.y_of(100.0)
        assert isinstance(y, float)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------
class TestXYDataNormalize:
    def test_normalize_max(self):
        d = _make(slope=1.0, offset=1.0)
        d.normalize()
        assert abs(max(d.y) - 1.0) < 1e-10

    def test_normalize_custom_val(self):
        d = _make(slope=1.0, offset=1.0)
        d.normalize(norm_val=2.0)
        assert abs(max(d.y) - 2.0) < 1e-10

    def test_normalize_with_xlim(self):
        d = _make(slope=1.0, offset=0.0, x0=0.0, x1=10.0)
        d.normalize(x_lim=(0.0, 5.0))
        # max within [0,5] normalised to 1 (y[5]=5, so 5/5=1)
        assert abs(d.y_of(5.0) - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Equidist
# ---------------------------------------------------------------------------
class TestXYDataEquidist:
    def test_uniform_grid(self):
        x = np.array([0.0, 1.0, 3.0, 6.0])
        y = np.array([0.0, 1.0, 3.0, 6.0])
        d = XYData(x, y.copy())
        d.equidist(left=0.0, right=6.0, delta=1.0, kind='linear')
        diffs = np.diff(d.x)
        assert np.allclose(diffs, diffs[0], atol=1e-10)

    def test_length_correct(self):
        d = _make()
        d.equidist(left=0, right=10, delta=0.5)
        assert len(d.x) == 21


# ---------------------------------------------------------------------------
# Savgol
# ---------------------------------------------------------------------------
class TestXYDataSavgol:
    def test_returns_copy(self):
        d = _make(n=100)
        sg = d.savgol(n1=11, n2=1)
        assert sg is not d

    def test_smooth_noisy(self):
        x = np.linspace(0, 10, 200)
        rng = np.random.default_rng(0)
        y = np.sin(x) + rng.normal(0, 0.1, 200)
        d = XYData(x, y)
        sg = d.savgol(n1=21, n2=2)
        # smoothed should have less std
        assert np.std(sg.y) < np.std(d.y)

    def test_with_name(self):
        d = _make(n=100)
        sg = d.savgol(n1=11, n2=1, name='smoothed')
        assert sg.name == 'smoothed'


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------
class TestXYDataResidual:
    def test_residual_zero_same_data(self):
        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        d1 = XYData(x.copy(), y.copy())
        d2 = XYData(x.copy(), y.copy())
        res = d1.residual(d2)
        np.testing.assert_allclose(res.y, 0.0, atol=1e-10)

    def test_residual_offset(self):
        x = np.linspace(0, 5, 50)
        d1 = XYData(x.copy(), np.ones(50) * 3.0)
        d2 = XYData(x.copy(), np.ones(50) * 1.0)
        res = d1.residual(d2)
        np.testing.assert_allclose(res.y, 2.0, atol=1e-10)

    def test_residual_relative(self):
        x = np.linspace(0, 5, 50)
        d1 = XYData(x.copy(), np.ones(50) * 4.0)
        d2 = XYData(x.copy(), np.ones(50) * 2.0)
        res = d1.residual(d2, relative=True)
        # (4-2)/4 = 0.5
        np.testing.assert_allclose(res.y, 0.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Chisquare
# ---------------------------------------------------------------------------
class TestXYDataChisquare:
    def test_chisquare_perfect_fit(self):
        x = np.linspace(1, 5, 50)
        y = np.ones(50) * 2.0
        d1 = XYData(x.copy(), y.copy())
        d2 = XYData(x.copy(), y.copy())
        chi2 = XYData.chisquare(d1, d2)
        assert abs(chi2) < 1e-10

    def test_chisquare_nonzero(self):
        x = np.linspace(1, 5, 50)
        d1 = XYData(x.copy(), np.ones(50) * 3.0)
        d2 = XYData(x.copy(), np.ones(50) * 2.0)
        chi2 = XYData.chisquare(d1, d2)
        assert chi2 > 0


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------
class TestXYDataDiff:
    def test_diff_linear(self):
        x = np.linspace(0, 5, 100)
        d = XYData(x, 3.0 * x + 1.0)
        result = d.diff()
        np.testing.assert_allclose(result.y[1:-1], 3.0, atol=0.01)

    def test_diff_with_range(self):
        x = np.linspace(0, 5, 100)
        d = XYData(x, 3.0 * x + 1.0)
        result = d.diff(left=1.0, right=4.0)
        assert min(result.x) >= 1.0 - 0.1
        assert max(result.x) <= 4.0 + 0.1


# ---------------------------------------------------------------------------
# Max / Min within
# ---------------------------------------------------------------------------
class TestXYDataMaxMin:
    def test_max_within_range(self):
        x = np.linspace(0, 10, 101)
        d = XYData(x, np.sin(x))
        m = d.max_within(0.0, 5.0)
        assert m <= 1.0 + 1e-10

    def test_min_within_range(self):
        x = np.linspace(0, 10, 101)
        d = XYData(x, -np.abs(x - 5))
        m = d.min_within(4.5, 5.5)
        assert m < 0.0

    def test_min_within_absolute(self):
        x = np.linspace(0, 5, 51)
        d = XYData(x, x - 2.0)
        m = d.min_within(absolute=True)
        assert m >= 0


# ---------------------------------------------------------------------------
# Cuts / zeros
# ---------------------------------------------------------------------------
class TestXYDataCuts:
    def test_cut_outside(self):
        x = np.linspace(0, 10, 101)
        d = XYData(x, x.copy())
        trimmed = d.cut_data_outside(2.0, 8.0)
        assert min(trimmed.x) >= 2.0 - 0.01
        assert max(trimmed.x) <= 8.0 + 0.01

    def test_zero_data(self):
        x = np.linspace(0, 10, 101)
        d = XYData(x, np.ones(101))
        result = d.zero_data(3.0, 7.0)
        # Values in [3,7] should be zero
        idx3 = int(3.0 / 10.0 * 100)
        idx7 = int(7.0 / 10.0 * 100)
        assert result.y[idx3] == 0.0
        # Values outside should still be 1
        assert result.y[0] == 1.0


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------
class TestXYDataShifts:
    def test_shift_x(self):
        d = _make()
        original = d.x.copy()
        d.shift_x(5.0)
        np.testing.assert_array_equal(d.x, original + 5.0)

    def test_shift_y(self):
        d = _make()
        original = d.y.copy()
        d.shift_y(-1.0)
        np.testing.assert_array_equal(d.y, original - 1.0)


# ---------------------------------------------------------------------------
# Remove NaN
# ---------------------------------------------------------------------------
class TestXYDataRemoveNan:
    def test_removes_nans(self):
        x = np.array([1.0, np.nan, 3.0])
        y = np.array([1.0, np.nan, 3.0])
        d = XYData(x, y, check_data=False)
        d.remove_nan()
        assert len(d.x) == 2
        assert not np.any(np.isnan(d.x))


# ---------------------------------------------------------------------------
# Polyfit
# ---------------------------------------------------------------------------
class TestXYDataPolyfit:
    def test_linear_fit(self):
        x = np.linspace(0, 10, 100)
        d = XYData(x, 2.0 * x + 3.0)
        # new_meshsize=0 → keep original x grid
        fit = d.polyfit(order=1, new_meshsize=0)
        np.testing.assert_allclose(fit.y, d.y, atol=1e-6)

    def test_quadratic_fit(self):
        x = np.linspace(0, 5, 100)
        d = XYData(x, x ** 2)
        fit = d.polyfit(order=2, new_meshsize=0)
        np.testing.assert_allclose(fit.y, x ** 2, atol=1e-6)


# ---------------------------------------------------------------------------
# Plot (non-crashing)
# ---------------------------------------------------------------------------
class TestXYDataPlot:
    def test_plot_no_crash(self):
        d = _make()
        plt.figure()
        d.plot(show_plot=False)
        plt.close('all')

    def test_plot_returns_fig(self):
        d = _make()
        fig = d.plot(show_plot=False, return_fig=True)
        assert fig is not None
        plt.close('all')

    def test_plot_with_hline_vline(self):
        d = _make()
        d.plot(show_plot=False, hline=5.0, vline=3.0)
        plt.close('all')

    def test_plot_log_scale(self):
        x = np.linspace(1, 10, 50)
        d = XYData(x, np.exp(x))
        d.plot(show_plot=False, yscale='log')
        plt.close('all')

    def test_bottom_top_for_plot(self):
        d = _make()
        bottom, top = d.bottom_top_for_plot(yscale='linear')
        assert bottom < top

    def test_bottom_top_log_scale(self):
        x = np.linspace(1, 5, 50)
        d = XYData(x, np.exp(x))
        bottom, top = d.bottom_top_for_plot(yscale='log')
        assert bottom > 0
        assert bottom < top


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------
class TestXYDataProduct:
    def test_product_ones(self):
        x = np.linspace(0, 5, 50)
        a = XYData(x, np.ones(50) * 3.0)
        b = XYData(x, np.ones(50) * 2.0)
        result = a.product(b)
        np.testing.assert_allclose(result.y, 6.0, atol=0.01)

    def test_product_with_labels(self):
        x = np.linspace(0, 5, 50)
        a = XYData(x, np.ones(50))
        b = XYData(x, np.ones(50) * 2.0)
        result = a.product(b, qy='Custom', uy='units')
        assert result.qy == 'Custom'
        assert result.uy == 'units'


# ---------------------------------------------------------------------------
# Monotoneous / strictly ascending
# ---------------------------------------------------------------------------
class TestXYDataSorting:
    def test_monotoneous_ascending(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        d = XYData(x, y, check_data=False)
        d.monotoneous_ascending()
        assert list(d.x) == [1.0, 2.0, 3.0]

    def test_strictly_ascending(self):
        x = np.array([0.0, 1.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 1.0, 2.0])
        d = XYData(x, y, check_data=False)
        d.strictly_ascending()
        assert len(d.x) == 3
        assert list(d.x) == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Reverse / swap_axes
# ---------------------------------------------------------------------------
class TestXYDataTransforms:
    def test_reverse(self):
        d = _make()
        original_x0 = d.x[0]
        d.reverse()
        assert d.x[0] == pytest.approx(10.0, abs=0.01)
        assert d.x[-1] == original_x0

    def test_swap_axes(self):
        d = _make(slope=2.0, offset=0.0)
        swapped = d.swap_axes()
        np.testing.assert_array_equal(swapped.x, d.y)
        np.testing.assert_array_equal(swapped.y, d.x)


# ---------------------------------------------------------------------------
# Idx range
# ---------------------------------------------------------------------------
class TestXYDataIdxRange:
    def test_full_range(self):
        d = _make()
        r = d.idx_range()
        assert len(r) == 50

    def test_partial_range(self):
        d = _make(x0=0, x1=10, n=101)
        r = d.idx_range(left=2.0, right=8.0)
        assert d.x[r[0]] >= 2.0 - 0.01
        assert d.x[r[-1]] <= 8.0 + 0.01


# ---------------------------------------------------------------------------
# MXYData
# ---------------------------------------------------------------------------
def _make_collection(n=3):
    items = []
    for i in range(n):
        x = np.linspace(0, 5, 50)
        y = float(i + 1) * x
        items.append(XYData(x.copy(), y.copy(), name=f'trace_{i}'))
    return MXYData(items)


class TestMXYData:
    def test_construction(self):
        m = _make_collection()
        assert m.n_y == 3

    def test_iterate(self):
        m = _make_collection()
        assert sum(1 for _ in m) == 3

    def test_label(self):
        m = _make_collection()
        m.label(['a', 'b', 'c'])
        assert m.label_defined is True
        assert m.lab == ['a', 'b', 'c']

    def test_copy(self):
        m = _make_collection()
        m.label(['a', 'b', 'c'])
        c = m.copy()
        assert c.n_y == 3
        assert c.lab == ['a', 'b', 'c']

    def test_append(self):
        m = _make_collection()
        extra = XYData(np.array([0.0, 1.0]), np.array([0.0, 1.0]), name='extra')
        m.append(extra)
        assert len(m.sa) == 4

    def test_combine(self):
        m1 = _make_collection()
        m2 = _make_collection()
        combined = MXYData.combine(m1, m2)
        assert len(combined.sa) == 6

    def test_generate_empty(self):
        m = MXYData.generate_empty()
        assert m.n_y == 0

    def test_mul_scalar(self):
        m = _make_collection()
        m2 = m * 2
        assert len(m2.sa) == 3

    def test_replace(self):
        m = _make_collection()
        new_sp = XYData(np.linspace(0, 5, 50), np.ones(50), name='replaced')
        m.replace(0, new_sp)
        assert m.sa[0].name == 'replaced'

    def test_delete(self):
        m = _make_collection()
        m.delete(m.sa[0])
        assert len(m.sa) == 2

    def test_qx_ux(self):
        m = _make_collection()
        m.qx_ux('Wavelength', 'nm')
        for sp in m.sa:
            assert sp.qx == 'Wavelength'
            assert sp.ux == 'nm'

    def test_qy_uy(self):
        m = _make_collection()
        m.qy_uy('Intensity', 'cps')
        for sp in m.sa:
            assert sp.qy == 'Intensity'
            assert sp.uy == 'cps'

    def test_names_to_label(self):
        m = _make_collection()
        m.names_to_label()
        assert m.label_defined is True
        assert m.lab[0] == 'trace_0'

    def test_remain(self):
        m = _make_collection(n=5)
        m.label(['a', 'b', 'c', 'd', 'e'])
        rem = m.remain([0, 2, 4])
        assert len(rem.sa) == 3

    def test_plot_no_crash(self):
        m = _make_collection()
        m.label(['a', 'b', 'c'])
        m.plot(show_plot=False)
        plt.close('all')

    def test_save_in_one_file(self):
        m = _make_collection()
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = os.path.join(tmpdir, 'combined.csv')
            m.save_in_one_file(fp)
            assert os.path.exists(fp)

    def test_load_individual(self):
        # Create fixture CSV files and load them
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                fname = f'test_{i}.csv'
                x = np.array([0.0, 1.0, 2.0])
                y = np.array([float(i), float(i)+1, float(i)+2])
                pd.DataFrame({'x': x, 'y': y}).to_csv(
                    os.path.join(tmpdir, fname), index=False)
            m = MXYData.load_individual(tmpdir)
            assert m.n_y == 3
