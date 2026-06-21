"""Fifth coverage-boost: XYdata branches, TRPL from_param branches, Spectrum methods."""
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# XYData — idfac_fit with plot=True and left/right=None
# ---------------------------------------------------------------------------
class TestXYDataIdfacFitPlot:
    def test_idfac_fit_plot_true(self):
        from fte_analysis_libraries.XYdata import XYData
        # JV semilog: Voc vs log(light intensity)
        light = np.array([10.0, 30.0, 50.0, 75.0, 100.0])
        nid = 1.5
        kTq = 0.02585
        Voc = nid * kTq * np.log10(light / 10) * np.log(10) + 0.8
        d = XYData(light, Voc)
        fit = d.idfac_fit(left=None, right=None, plot=True, return_fit=True)
        plt.close('all')
        assert fit is not None

    def test_idfac_fit_left_right_clamp(self):
        from fte_analysis_libraries.XYdata import XYData
        light = np.array([10.0, 30.0, 50.0, 75.0, 100.0])
        nid = 1.5
        kTq = 0.02585
        Voc = nid * kTq * np.log10(light / 10) * np.log(10) + 0.8
        d = XYData(light, Voc)
        # left=None → left gets set to min(self.x); right=None → max(self.x)
        fit = d.idfac_fit(left=None, right=None, return_fit=True, plot=False)
        assert hasattr(fit, 'nid')

    def test_plot_linfit_return_data(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        d = XYData(x, y)
        result = d.plot_linfit(residue=False, return_data=True)
        plt.close('all')
        assert result is not None

    def test_plot_with_create_image_stream(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        d = XYData(x, np.sin(x))
        stream = d.plot(show_plot=False, create_image_stream=True)
        plt.close('all')
        assert stream is not None


# ---------------------------------------------------------------------------
# MXYData — combine, set_plotstyle branches, print_all_names
# ---------------------------------------------------------------------------
class TestMXYDataUtilities:
    def _make_m(self, n=3):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        return MXYData([XYData(x, np.ones(50) * (i+1), name=f'trace_{i}')
                        for i in range(n)])

    def test_combine_labeled(self):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m1 = MXYData([XYData(x, np.ones(50), name='a')])
        m2 = MXYData([XYData(x, np.ones(50)*2, name='b')])
        m1.label(['A'])
        m2.label(['B'])
        combined = MXYData.combine(m1, m2)
        assert len(combined.sa) == 2
        assert combined.label_defined

    def test_combine_different_types_warning(self):
        """Combining different types prints warning but doesn't raise."""
        from fte_analysis_libraries.Spectrum import Spectra, Spectrum
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(400, 700, 50)
        m1 = MXYData([XYData(x, np.ones(50), name='a')])
        m2 = Spectra([Spectrum(x, np.ones(50)*2, name='b')])
        # Different types → should still work (prints warning)
        combined = MXYData.combine(m1, m2)
        assert len(combined.sa) == 2

    def test_set_plotstyle_all_params(self):
        m = self._make_m()
        m.set_plotstyle(marker='o', markersize=10, linewidth=2.0, color='red')
        for sp in m.sa:
            assert sp.plotstyle['marker'] == 'o'
            assert sp.plotstyle['markersize'] == 10
            assert sp.plotstyle['linewidth'] == 2.0

    def test_print_all_names_split(self):
        m = self._make_m()
        m.print_all_names(split_ch='_')

    def test_print_all_names_unique(self):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([
            XYData(x, np.ones(50), name='group_A'),
            XYData(x, np.ones(50)*2, name='group_A'),
            XYData(x, np.ones(50)*3, name='group_B'),
        ])
        m.print_all_names(unique_only=True, split_ch='_', print_idx=False)

    def test_print_all_names_return_list(self):
        m = self._make_m()
        result = m.print_all_names(return_list=True, print_all=False)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_print_names_containing(self):
        m = self._make_m()
        m.print_names_containing('trace_1')

    def test_plot_showindex(self):
        m = self._make_m(2)
        m.label(['first', 'second'])
        m.plot(show_plot=False, showindex=True)
        plt.close('all')

    def test_plot_individual_style(self):
        m = self._make_m(2)
        m.label(['first', 'second'])
        m.plot(show_plot=False, plotstyle='individual')
        plt.close('all')

    def test_plot_ylabel_xlabel(self):
        m = self._make_m(2)
        m.plot(show_plot=False, ylabel='y-axis', xlabel='x-axis')
        plt.close('all')

    def test_plot_hline_mxy(self):
        m = self._make_m(2)
        m.plot(show_plot=False, hline=2.0, vline=2.5)
        plt.close('all')

    def test_names_to_label_split(self):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.ones(50), name='sample_A_001'),
                     XYData(x, np.ones(50)*2, name='sample_B_002')])
        m.names_to_label(split_ch='_')
        assert m.label_defined


# ---------------------------------------------------------------------------
# Spectrum — AbsSpectrum.new_ue with correct args
# ---------------------------------------------------------------------------
class TestAbsSpectrumNewUe:
    def test_new_ue_modifies_inplace(self):
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        eV = np.linspace(1.0, 3.5, 300)
        absorptance = np.clip(1 / (1 + np.exp(-(eV - 1.8) * 10)), 0, 1)
        sp = AbsSpectrum(eV, absorptance)
        original_y = sp.y.copy()
        sp.new_ue(UE=30.0, E_takeover=1.5)
        # Below E_takeover region should be modified
        assert not np.allclose(sp.y, original_y)


# ---------------------------------------------------------------------------
# TRPL.from_param — show_progress and normalize_ns
# ---------------------------------------------------------------------------
class TestTRPLFromParamBranches:
    def _make_params(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        return TRPLParam(dt=1e-12, finaltime=1e-9, thickness=100,
                         N_points=10, k1=1e7, k2=1e-10, pulse_len=None)

    def test_from_param_show_progress(self):
        import numpy as np

        from fte_analysis_libraries.TRPL import TRPLData, TRPLParam
        p = self._make_params()
        p.n0 = np.ones(p.N_points) * 1e15
        d = TRPLData.from_param(p, time_delta=0.1e-9, show_progress=True)
        assert d.y.max() == 1.0

    def test_from_param_normalize_ns(self):
        import numpy as np

        from fte_analysis_libraries.TRPL import TRPLData, TRPLParam
        p = self._make_params()
        p.n0 = np.ones(p.N_points) * 1e15
        d = TRPLData.from_param(p, time_delta=0.1e-9,
                                 normalize_ns=0.2, normalize_cts=100.0)
        # normalize_ns branch: data.y = data.y * normalize_cts / data.y_of(normalize_ns)
        assert d is not None


# ---------------------------------------------------------------------------
# XYData — load_individual with take_quants_and_units_from_file
# ---------------------------------------------------------------------------
class TestMXYDataLoadIndividual:
    def test_load_individual_with_quants(self):
        import os
        import tempfile

        from fte_analysis_libraries.XYdata import MXYData, XYData
        with tempfile.TemporaryDirectory() as tmp:
            import pandas as pd
            for i in range(2):
                x = np.linspace(0, 5, 20)
                y = np.sin(x) + i
                df = pd.DataFrame({
                    'Wavelength (nm)': x,
                    'Intensity (counts)': y
                })
                df.to_csv(os.path.join(tmp, f'trace_{i}.csv'), index=False)
            m = MXYData.load_individual(tmp, take_quants_and_units_from_file=True)
            assert len(m.sa) == 2
            assert m.sa[0].qx == 'Wavelength'
            assert m.sa[0].ux == 'nm'


# ---------------------------------------------------------------------------
# MXYData — plot with generate_image_stream
# ---------------------------------------------------------------------------
class TestMXYDataImageStream:
    def test_plot_generate_image_stream(self):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.ones(50), name='a'),
                     XYData(x, np.ones(50)*2, name='b')])
        m.label(['A', 'B'])
        stream = m.plot(show_plot=False, generate_image_stream=True)
        plt.close('all')
        # generate_image_stream returns an image stream


# ---------------------------------------------------------------------------
# IV — MIVData construction and methods
# ---------------------------------------------------------------------------
class TestMIVData:
    def _make_ivs(self, n=2):
        from fte_analysis_libraries.IV import IVData
        ivs = []
        for i in range(n):
            V = np.linspace(-0.1, 1.0, 200)
            iv = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5+i*0.2,
                                 Rs=2.0, Rsh=5000.0, light_int=100.0)
            iv.det_perfparam(minimal=True)
            ivs.append(iv)
        return ivs

    def test_miv_construction(self):
        from fte_analysis_libraries.IV import MIVData
        ivs = self._make_ivs()
        m = MIVData(ivs)
        assert len(m.sa) == 2

    def test_miv_plot(self):
        from fte_analysis_libraries.IV import MIVData
        ivs = self._make_ivs()
        m = MIVData(ivs)
        m.label(['iv1', 'iv2'])
        m.plot(show_plot=False)
        plt.close('all')


# ---------------------------------------------------------------------------
# TRPL — dlifetime with x='qfls'
# ---------------------------------------------------------------------------
class TestTRPLDlifetime:
    def test_dlifetime_default(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 500, 501)
        y = 1000.0 * np.exp(-t / 150.0)
        dat = TRPLData(t, y)
        dl = dat.dlifetime(fluence=5e-9)
        assert len(dl.x) > 0

    def test_dlifetime_qfls(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 500, 501)
        y = 1000.0 * np.exp(-t / 150.0)
        dat = TRPLData(t, y)
        dl = dat.dlifetime(fluence=5e-9, x='qfls', wavelength=532,
                           film_thickness=500, ni=1e10)
        assert len(dl.x) > 0
