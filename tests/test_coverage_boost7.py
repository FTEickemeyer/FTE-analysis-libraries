"""Seventh coverage-boost: TRPL auto-detect branches, k1k2_model_fit, XYdata edges."""
import os
import tempfile
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# TRPL helpers: _trpl_prepare with start/stop=None and used_for_fit branch
# ---------------------------------------------------------------------------
class TestTRPLPrepare:
    def _make_bimol(self, k1=1e6, k2=1e-11, n0=1e14, t_max=400.0):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, t_max, 401)
        y = k1/k2 * 1/(np.exp(k1*t*1e-9) * (1 + k1/(n0*k2)) - 1)
        return TRPLData(t, y / y[0])

    def test_trpl_prepare_start_none_stop_none(self):
        """_trpl_prepare with start=None → 0, stop=None → x[-1]."""
        dat = self._make_bimol()
        start, stop, d, r, n0 = dat._trpl_prepare(None, None, 'savgol', None)
        assert start == 0
        assert stop == dat.x[-1]

    def test_trpl_prepare_used_for_fit_not_savgol(self):
        """_trpl_prepare with used_for_fit != 'savgol' → dat = self (line 1230)."""
        dat = self._make_bimol()
        start, stop, d, r, n0 = dat._trpl_prepare(0, 300, 'raw', None)
        assert d is dat  # no savgol smoothing: returns self

    def test_k1_k2_fit_show_all(self):
        dat = self._make_bimol()
        result, popt = dat.k1_k2_fit(start=0, stop=300, show_all=True)
        plt.close('all')
        assert len(popt) == 3


# ---------------------------------------------------------------------------
# TRPL: k1_k2_model_fit with what_to_fit branches
# ---------------------------------------------------------------------------
class TestTRPLK1K2ModelFit:
    def _make_bimol(self, k1=1e6, k2=1e-11, n0=1e14, t_max=400.0):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, t_max, 401)
        y = k1/k2 * 1/(np.exp(k1*t*1e-9) * (1 + k1/(n0*k2)) - 1)
        return TRPLData(t, y / y[0])

    def test_model_fit_no_fit(self):
        """what_to_fit=None → only n0 is fit."""
        dat = self._make_bimol()
        PL_dta, [n0, k1, k2] = dat.k1_k2_model_fit(
            start=0, stop=300, n0=1e14, k1=1e6, k2=1e-11, what_to_fit=None)
        assert n0 > 0
        assert k1 == 1e6  # unchanged
        assert k2 == 1e-11  # unchanged

    def test_model_fit_k1(self):
        """what_to_fit=['k1'] → fit k1 only."""
        dat = self._make_bimol()
        PL_dta, [n0, k1, k2] = dat.k1_k2_model_fit(
            start=0, stop=300, n0=1e14, k1=1e6, k2=1e-11, what_to_fit=['k1'])
        assert k1 > 0

    def test_model_fit_k2(self):
        """what_to_fit=['k2'] → fit k2 only."""
        dat = self._make_bimol()
        PL_dta, [n0, k1, k2] = dat.k1_k2_model_fit(
            start=0, stop=300, n0=1e14, k1=1e6, k2=1e-11, what_to_fit=['k2'])
        assert k2 > 0

    def test_model_fit_k1_k2(self):
        """what_to_fit=['k1', 'k2'] → fit both."""
        dat = self._make_bimol()
        PL_dta, [n0, k1, k2] = dat.k1_k2_model_fit(
            start=0, stop=300, n0=1e14, k1=1e6, k2=1e-11, what_to_fit=['k1', 'k2'])
        assert k1 > 0
        assert k2 > 0

    def test_model_fit_with_show(self):
        """show='all' exercises all four show branches."""
        dat = self._make_bimol()
        PL_dta, params = dat.k1_k2_model_fit(
            start=0, stop=200, n0=1e14, k1=1e6, k2=1e-11,
            what_to_fit=['k1'], show='all')
        plt.close('all')
        assert PL_dta is not None


# ---------------------------------------------------------------------------
# TRPL: del_bg with auto-detect (no start/stop), plot_details, norm_val
# ---------------------------------------------------------------------------
class TestTRPLDelBgAuto:
    def _make_pulse(self):
        """Make a synthetic TRPL trace with rising edge + exponential decay."""
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 500, 501)
        # Rising edge from 0 to peak at t=50, then exponential decay
        y = np.zeros(501)
        y[:50] = np.linspace(0, 1000, 50)  # rising
        y[50:] = 1000.0 * np.exp(-(t[50:] - t[50]) / 150.0)
        y[:10] = 5.0  # background noise before pulse
        return TRPLData(t, y)

    def test_del_bg_auto_detect(self):
        """del_bg() with start=None, stop=None uses inner auto-detect functions."""
        dat = self._make_pulse()
        result = dat.del_bg()
        assert result is not None

    def test_del_bg_auto_with_norm(self):
        """del_bg with norm_val → normalize after background subtraction."""
        dat = self._make_pulse()
        result = dat.del_bg(norm_val=1.0)
        assert abs(result.y.max() - 1.0) < 0.1

    def test_del_bg_plot_details(self):
        """del_bg with plot_details=True exercises lines 1482-1491."""
        dat = self._make_pulse()
        result = dat.del_bg(start=5.0, stop=45.0, plot_details=True)
        plt.close('all')
        assert result is not None

    def test_del_bg_plot_details_with_norm(self):
        """del_bg with plot_details=True and norm_val set covers lines 1487-1489."""
        dat = self._make_pulse()
        result = dat.del_bg(start=5.0, stop=45.0, norm_val=1.0, plot_details=True)
        plt.close('all')
        assert abs(result.y.max() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# TRPL: shift_to_max with plot_details=True (lines 1532-1540)
# ---------------------------------------------------------------------------
class TestTRPLShiftToMaxPlotDetails:
    def test_shift_to_max_plot_details(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(-50, 500, 551)
        y = np.exp(-((t - 100) ** 2) / 500) + 0.001
        dat = TRPLData(t, y)
        result = dat.shift_to_max(plot_details=True, left=0, right=400)
        plt.close('all')
        assert result.x[np.argmax(result.y)] == pytest.approx(0.0, abs=1.0)

    def test_shift_to_max_plot_details_no_limits(self):
        """left=None, right=None → auto-detected (lines 1532-1535)."""
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 500, 501)
        y = np.exp(-((t - 150) ** 2) / 1000) + 0.001
        dat = TRPLData(t, y)
        result = dat.shift_to_max(plot_details=True, left=None, right=None)
        plt.close('all')
        assert result is not None


# ---------------------------------------------------------------------------
# TRPL: MTRPLData.load_individual (lines 1573-1604)
# ---------------------------------------------------------------------------
class TestMTRPLDataLoadIndividual:
    def test_load_individual_basic(self):
        import pandas as pd

        from fte_analysis_libraries.TRPL import MTRPLData
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                t = np.linspace(0, 500, 501)
                y = 1000.0 * np.exp(-t / (100 + i * 50))
                df = pd.DataFrame({'Time (ns)': t, 'PL (cts)': y})
                df.to_csv(os.path.join(tmp, f'trace_{i}.csv'), index=False)
            m = MTRPLData.load_individual(tmp)
            assert len(m.sa) == 2

    def test_load_individual_with_quants(self):
        import pandas as pd

        from fte_analysis_libraries.TRPL import MTRPLData
        with tempfile.TemporaryDirectory() as tmp:
            t = np.linspace(0, 300, 301)
            y = np.exp(-t / 80)
            df = pd.DataFrame({'Time (ns)': t, 'PL intensity (cts)': y})
            df.to_csv(os.path.join(tmp, 'trace.csv'), index=False)
            m = MTRPLData.load_individual(tmp, take_quants_and_units_from_file=True)
            assert m.sa[0].qx == 'Time'
            assert m.sa[0].ux == 'ns'


# ---------------------------------------------------------------------------
# TRPL: MTRPLData._batch_expfit showparam=True (lines 1611, 1616-1620)
# ---------------------------------------------------------------------------
class TestMTRPLBatchShowparam:
    def _make_batch(self, n=2):
        from fte_analysis_libraries.TRPL import MTRPLData, TRPLData
        taus = [80.0, 150.0]
        sa = []
        for tau in taus[:n]:
            t = np.linspace(0, 500, 501)
            y = np.exp(-t / tau)
            sa.append(TRPLData(t, y, name=f'trace_tau{tau:.0f}'))
        return MTRPLData(sa)

    def test_batch_mono_expfit_showparam(self):
        m = self._make_batch()
        result = m.mono_expfit(start=0, stop=400, showparam=True)
        plt.close('all')
        assert len(result.sa) == 2

    def test_batch_mult2_expfit_showparam(self):
        from fte_analysis_libraries.TRPL import MTRPLData, TRPLData
        t = np.linspace(0, 500, 501)
        sa = [TRPLData(t, 0.7*np.exp(-t/80) + 0.3*np.exp(-t/250), name=f'tr_{i}')
              for i in range(2)]
        m = MTRPLData(sa)
        result = m.mult2_expfit(start=0, stop=400, showparam=True)
        plt.close('all')
        assert len(result.sa) == 2


# ---------------------------------------------------------------------------
# XYData/MXYData: _bottom_top_for_plot with log scale and bottom=0
# ---------------------------------------------------------------------------
class TestBottomTopForPlotDivisor:
    def test_xy_bottom_top_log_zero_no_divisor(self):
        """Lines 59-61: bottom=0 on log scale → standard divisor 1e8."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        y = np.zeros(50)
        y[10:] = np.exp(-x[10:])
        d = XYData(x, y)
        bottom, top = d.bottom_top_for_plot(yscale='log')
        assert bottom > 0

    def test_xy_bottom_top_log_zero_with_divisor(self):
        """Lines 62-63: bottom=0 on log scale + divisor → bottom = top/divisor."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        y = np.zeros(50)
        y[5:] = np.exp(-x[5:])
        d = XYData(x, y)
        bottom, top = d.bottom_top_for_plot(yscale='log', divisor=1e4)
        assert abs(bottom - top / 1e4) < 1e-10

    def test_mxy_bottom_top_log_zero_no_divisor(self):
        """MXYData.bottom_top_for_plot lines 2348-2351: no divisor."""
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.zeros(50)), XYData(x, np.exp(-x))])
        bottom, top = m.bottom_top_for_plot(yscale='log')
        assert bottom > 0

    def test_mxy_bottom_top_log_zero_with_divisor(self):
        """MXYData.bottom_top_for_plot lines 2352-2353: with divisor."""
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.zeros(50)), XYData(x, np.exp(-x))])
        bottom, top = m.bottom_top_for_plot(yscale='log', divisor=500.0)
        assert abs(bottom - top / 500.0) < 1e-10


# ---------------------------------------------------------------------------
# XYZData: load from file (lines 2688-2720)
# ---------------------------------------------------------------------------
class TestXYZDataLoad:
    def test_load_basic(self):
        import pandas as pd

        from fte_analysis_libraries.XYdata import XYZData
        with tempfile.TemporaryDirectory() as tmp:
            x = np.linspace(0, 5, 20)
            df = pd.DataFrame({'x': x, 'y': np.sin(x), 'z': np.cos(x)})
            fp = os.path.join(tmp, 'data.csv')
            df.to_csv(fp, index=False)
            obj = XYZData.load(tmp, 'data.csv')
            assert len(obj.x) == 20

    def test_load_with_quants_from_file(self):
        import pandas as pd

        from fte_analysis_libraries.XYdata import XYZData
        with tempfile.TemporaryDirectory() as tmp:
            x = np.linspace(0, 5, 10)
            df = pd.DataFrame({
                'Voltage (V)': x,
                'Current (mA)': np.ones(10),
                'Power (mW)': x * 2.0
            })
            fp = os.path.join(tmp, 'data.csv')
            df.to_csv(fp, index=False)
            obj = XYZData.load(tmp, 'data.csv', take_quants_and_units_from_file=True)
            assert obj.qx == 'Voltage'
            assert obj.ux == 'V'

    def test_load_first_file_in_dir(self):
        """load() with empty filepath → uses first file in directory."""
        import pandas as pd

        from fte_analysis_libraries.XYdata import XYZData
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0], 'z': [5.0, 6.0]})
            df.to_csv(os.path.join(tmp, 'first.csv'), index=False)
            obj = XYZData.load(tmp, filepath='')
            assert len(obj.x) == 2


# ---------------------------------------------------------------------------
# MXYData.plot: create_image_stream and save_plot branches (lines 2091-2112)
# ---------------------------------------------------------------------------
class TestMXYDataPlotBranches:
    def _make_m(self, n=2):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        return MXYData([XYData(x, np.sin(x + i), name=f'trace_{i}') for i in range(n)])

    def test_plot_create_image_stream(self):
        """Lines 2107-2112: create_image_stream=True in MXYData.plot."""
        m = self._make_m()
        m.label(['A', 'B'])
        stream = m.plot(show_plot=False, create_image_stream=True)
        plt.close('all')
        assert stream is not None

    def test_plot_generate_image_stream(self):
        """generate_image_stream=True in MXYData.plot (different kwarg name)."""
        m = self._make_m()
        m.label(['A', 'B'])
        m.plot(show_plot=False, generate_image_stream=True)
        plt.close('all')

    def test_plot_title(self):
        """title != '' path (line 2087-2088)."""
        m = self._make_m()
        m.plot(show_plot=False, title='Test Title')
        plt.close('all')
