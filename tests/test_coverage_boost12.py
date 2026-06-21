"""Twelfth coverage-boost: PLQY laser branches, General utilities, Spectrum conversions, XYData edges."""
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
# PLQY.py — ExpParam with explicit excitation_laser (lines 432-466)
# ---------------------------------------------------------------------------
class TestExpParamLaserBranches:
    def test_laser_403(self):
        """Lines 432-433, 451-452: laser=403 → laser_left=390, marker='405BPF'."""
        from fte_analysis_libraries.PLQY import ExpParam
        p = ExpParam(excitation_laser=403)
        assert p.laser_left == 390
        assert p.laser_right == 420
        assert p.laser_marker == '405BPF'
        assert p.PL_marker == '450LPF'

    def test_laser_419(self):
        """Lines 435-436, 454-455: laser=419 → laser_left=410, marker='420BPF'."""
        from fte_analysis_libraries.PLQY import ExpParam
        p = ExpParam(excitation_laser=419)
        assert p.laser_left == 410
        assert p.laser_right == 428
        assert p.laser_marker == '420BPF'

    def test_laser_422(self):
        """Lines 441-442, 460-461: laser=422 → laser_left=416, marker='420BPF'."""
        from fte_analysis_libraries.PLQY import ExpParam
        p = ExpParam(excitation_laser=422)
        assert p.laser_left == 416
        assert p.laser_right == 426
        assert p.laser_marker == '420BPF'

    def test_laser_invalid(self):
        """Invalid laser → prints warning (line 447, 468)."""
        from fte_analysis_libraries.PLQY import ExpParam
        p = ExpParam(excitation_laser=999)
        assert p.excitation_laser == 999


# ---------------------------------------------------------------------------
# General.py — save_ok with quitted=True (lines 480, 485)
# ---------------------------------------------------------------------------
class TestSaveOkQuitted:
    def test_save_ok_quitted_true(self):
        """save_ok with quitted=True → ok=False immediately (line 480, 485)."""
        from fte_analysis_libraries.General import save_ok
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'test.csv')
            ok, quitted = save_ok(fp, quitted=True)
            assert ok is False
            assert quitted is True

    def test_save_ok_file_not_exist(self):
        """save_ok with non-existent file → returns True immediately (line 478, 483)."""
        from fte_analysis_libraries.General import save_ok
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'nonexistent.csv')
            ok = save_ok(fp)
            assert ok is True


# ---------------------------------------------------------------------------
# General.py — plot_first_n_lines (lines 586-598)
# ---------------------------------------------------------------------------
class TestPlotFirstNLines:
    def test_plot_first_n_lines(self):
        """Read first N lines from a text file (lines 586-598)."""
        from fte_analysis_libraries.General import plot_first_n_lines
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, 'sample.txt')
            with open(fp, 'w') as f:
                for i in range(30):
                    f.write(f'line {i}: some data {i*2.0}\n')
            # Call with n=5 → reads first 5 lines then breaks
            plot_first_n_lines(tmp, 'sample.txt', n=5)


# ---------------------------------------------------------------------------
# General.py — fullprint (lines 604-608)
# ---------------------------------------------------------------------------
class TestFullprint:
    def test_fullprint(self):
        """fullprint prints array without truncation (lines 604-608)."""
        from fte_analysis_libraries.General import fullprint
        arr = np.arange(500)
        fullprint(arr)  # should not raise


# ---------------------------------------------------------------------------
# Spectrum.py — conversion error branches (lines 732-789)
# ---------------------------------------------------------------------------
class TestSpectrumConversionErrors:
    def test_photonflux_to_irradiance_wrong_x_unit(self):
        """Lines 732-733: photonflux_to_irradiance with ux != 'nm'."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        eV = np.linspace(1.5, 3.0, 100)
        sp = DiffSpectrum(eV, np.ones(100),
                          units={'x': 'eV', 'y': '1/[s m2 nm]'})
        result = sp.photonflux_to_irradiance()
        assert result is None

    def test_irradiance_to_photonflux_wrong_x_unit(self):
        """Lines 762-763: wrong x unit when uy is correct."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        eV = np.linspace(1.5, 3.0, 100)
        sp = DiffSpectrum(eV, np.ones(100),
                          units={'x': 'eV', 'y': 'W/[m2 nm]'})
        result = sp.irradiance_to_photonflux()
        assert result is None

    def test_irradiance_to_photonflux_wrong_y_unit(self):
        """Lines 765-766: wrong y unit → prints warning."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 100)
        sp = DiffSpectrum(wl, np.ones(100),
                          units={'x': 'nm', 'y': 'WRONG'})
        result = sp.irradiance_to_photonflux()
        assert result is None

    def test_irradiance_to_illuminance_wrong_unit(self):
        """Lines 776-777: irradiance_to_illuminance with wrong uy."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(400, 720, 100)
        sp = DiffSpectrum(wl, np.ones(100),
                          units={'x': 'nm', 'y': '1/[s m2 nm]'})
        result = sp.irradiance_to_illuminance()
        assert result is None

    def test_illuminance_to_irradiance_with_warning(self):
        """Lines 787-789: illuminance_to_irradiance with warning=True (default)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        illum = am.irradiance_to_illuminance()
        irr = illum.illuminance_to_irradiance(warning=True)
        assert irr is not None


# ---------------------------------------------------------------------------
# Spectrum.py — EQESpectrum.load_cicci (line 378), AbsSpectrum.load_absorbance (643)
# ---------------------------------------------------------------------------
class TestSpectrumLoaderMethods:
    def test_eqespectrum_load_cicci(self):
        """Line 378: load_cicci uses tab delimiter."""
        import pandas as pd

        from fte_analysis_libraries.Spectrum import EQESpectrum
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(300, 800, 50)
            eqe = np.linspace(0, 80, 50)
            df = pd.DataFrame({'Wavelength': wl, 'EQE': eqe})
            fp = os.path.join(tmp, 'cicci.dat')
            df.to_csv(fp, sep='\t', index=False, header=True)
            sp = EQESpectrum.load_cicci(tmp, 'cicci.dat')
            assert len(sp.x) == 50

    def test_absorbance_load(self):
        """Line 643: AbsSpectrum.load_absorbance with CSV file."""
        import pandas as pd

        from fte_analysis_libraries.Spectrum import AbsSpectrum
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(400, 800, 50)
            ab = np.ones(50) * 0.5
            df = pd.DataFrame({'Wavelength (nm)': wl, 'Absorbance': ab})
            fp = os.path.join(tmp, 'abs.csv')
            df.to_csv(fp, index=False)
            sp = AbsSpectrum.load_absorbance(tmp, 'abs.csv')
            assert len(sp.x) == 50


# ---------------------------------------------------------------------------
# Spectrum.py — AbsSpectrum.calc_vocrad without Jradlim (line 514)
# ---------------------------------------------------------------------------
class TestAbsSpectrumCalcVocrad:
    def test_calc_vocrad_no_jradlim(self):
        """Line 514: AttributeError when Jradlim not set."""
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        E = np.linspace(1.5, 3.0, 100)
        ab = AbsSpectrum(E, np.ones(100) * 0.9,
                         quants={'x': 'Photon energy', 'y': 'Absorptance'},
                         units={'x': 'eV', 'y': ''})
        with pytest.raises((AttributeError, TypeError)):
            ab.calc_vocrad(E_start=1.6, E_stop=2.8)


# ---------------------------------------------------------------------------
# XYData — df_to_xy with wrong type (lines 234-235)
# ---------------------------------------------------------------------------
class TestDfToXYWrongType:
    def test_df_to_xy_wrong_type(self):
        """Lines 234-235: df is not DataFrame or Series → prints warning, returns None."""
        import pandas as pd

        from fte_analysis_libraries.XYdata import XYData
        result = XYData.from_df("not a dataframe")  # type: ignore
        assert result is None

    def test_df_to_xy_no_unit_in_index(self):
        """Line 253: ux = None when index name has no '('."""
        import pandas as pd

        from fte_analysis_libraries.XYdata import XYData
        df = pd.DataFrame({'y': [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        df.index.name = 'Time'  # no '(' → ux = None
        sp = XYData.from_df(df, take_quants_and_units_from_df=True)
        assert sp is not None


# ---------------------------------------------------------------------------
# XYData.x_of — interpolate=True branch (lines 416-432)
# ---------------------------------------------------------------------------
class TestXOfInterpolate:
    def test_x_of_interpolate(self):
        """Lines 416-432: x_of with interpolate=True (cubic interp)."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 100)
        y = x ** 2  # monotone
        sp = XYData(x, y)
        # Find x where y = 25 → should be 5.0
        result = sp.x_of(25.0, interpolate=True)
        assert abs(result - 5.0) < 0.1

    def test_x_of_interpolate_with_start(self):
        """Line 414: x_of with start set, then interpolate=True."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 100)
        y = x ** 2
        sp = XYData(x, y)
        result = sp.x_of(25.0, start=3.0, interpolate=True)
        assert abs(result - 5.0) < 0.1


# ---------------------------------------------------------------------------
# XYData.zero_data_outside — left/right=None (lines 1075, 1077)
# ---------------------------------------------------------------------------
class TestZeroData:
    def test_zero_data_no_limits(self):
        """Lines 1075, 1077: zero_data with left/right=None → uses min/max of x."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 50)
        y = np.ones(50)
        sp = XYData(x, y)
        result = sp.zero_data(left=None, right=None)
        # After zeroing everything, all y should be 0
        assert np.all(result.y == 0)

    def test_zero_data_limits_outside_range(self):
        """left < min(x) → l = min(x); right > max(x) → r = max(x)."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 50)
        y = np.ones(50)
        sp = XYData(x, y)
        result = sp.zero_data(left=-5.0, right=20.0)
        assert np.all(result.y == 0)


# ---------------------------------------------------------------------------
# XYData.cut_data_outside — left/right=None (lines 1094, 1096)
# ---------------------------------------------------------------------------
class TestCutDataOutside:
    def test_cut_data_outside_no_limits(self):
        """Lines 1094, 1096: left/right=None → uses min/max of x."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        sp = XYData(x, y)
        result = sp.cut_data_outside(left=None, right=None)
        assert len(result.x) == len(x)

    def test_cut_data_outside_limits_outside_range(self):
        """left < min(x) and right > max(x) → clamps to x range."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(1, 9, 50)
        y = np.sin(x)
        sp = XYData(x, y)
        result = sp.cut_data_outside(left=0.0, right=100.0)
        assert len(result.x) > 0


# ---------------------------------------------------------------------------
# XYData.polyfit — new_x_arr and new_meshsize branches (lines 1371, 1375)
# ---------------------------------------------------------------------------
class TestPolyfitBranches:
    def test_polyfit_new_meshsize(self):
        """Line 1371: polyfit with new_meshsize > 0."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        y = x ** 2 + 0.1 * np.random.randn(50)
        sp = XYData(x, y)
        fit = sp.polyfit(order=2, new_meshsize=200)
        assert len(fit.x) == 200

    def test_polyfit_new_x_arr(self):
        """Line 1375: polyfit with explicit new_x_arr."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        y = x ** 2
        sp = XYData(x, y)
        new_x = np.linspace(1, 4, 30)
        fit = sp.polyfit(order=2, new_x_arr=new_x)
        assert len(fit.x) == 30


# ---------------------------------------------------------------------------
# XYData.despike — near edge (lines 1447, 1451)
# ---------------------------------------------------------------------------
class TestRmCosrayEdges:
    def test_rm_cosray_spike_at_left_edge(self):
        """Line 1451: m_left = i when spike is at index < m."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 50)
        y = np.sin(x).copy()
        y[1] = 100.0  # spike at index 1 (< m=3)
        sp = XYData(x, y)
        result = sp.rm_cosray()
        assert abs(result.y[1]) < 50  # spike reduced

    def test_rm_cosray_spike_at_right_edge(self):
        """Line 1447: m_right = l_spikes - i - 1 when spike is near right edge."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 50)
        y = np.sin(x).copy()
        y[48] = 100.0  # spike near right boundary
        sp = XYData(x, y)
        result = sp.rm_cosray()
        assert abs(result.y[48]) < 50


# ---------------------------------------------------------------------------
# XYData.plot — ax parameter (line 546), bottom-only (565), top-only (571)
# ---------------------------------------------------------------------------
class TestXYDataPlotBranches:
    def test_plot_with_ax_parameter(self):
        """Line 546: passing ax sets show_plot=False."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        sp = XYData(x, np.sin(x))
        fig, ax = plt.subplots()
        sp.plot(ax=ax, show_plot=False)
        plt.close('all')

    def test_plot_bottom_only(self):
        """Line 565: bottom set but top=None → auto top from data."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        sp = XYData(x, np.exp(-x))
        sp.plot(show_plot=False, bottom=0.01)
        plt.close('all')

    def test_plot_top_only(self):
        """Line 571: top set but bottom=None → auto bottom from data."""
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        sp = XYData(x, np.exp(-x))
        sp.plot(show_plot=False, top=2.0)
        plt.close('all')


# ---------------------------------------------------------------------------
# MXYData.plot — line 2012 (left update) and 2066 (nolabel + plotstyle)
# ---------------------------------------------------------------------------
class TestMXYDataRemainingBranches:
    def test_plot_left_update_when_later_spec_starts_earlier(self):
        """Line 2012: second spec starts before first → left updated."""
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x1 = np.linspace(2, 8, 50)   # starts at 2
        x2 = np.linspace(0, 6, 50)   # starts at 0 < 2 → triggers line 2012
        m = MXYData([XYData(x1, np.sin(x1), name='a'),
                     XYData(x2, np.cos(x2), name='b')])
        m.label(['A', 'B'])
        m.plot(show_plot=False)
        plt.close('all')

    def test_plot_nolabel_individual_plotstyle(self):
        """Line 2066: nolabel=True + plotstyle='individual' → ax.plot with **plotstyle."""
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.sin(x), name='a'), XYData(x, np.cos(x), name='b')])
        m.label(['A', 'B'])
        m.set_plotstyle(linewidth=2.0)
        m.plot(show_plot=False, plotstyle='individual', nolabel=True)
        plt.close('all')
