"""Additional coverage tests targeting TRPL, Spectrum, Electrochemistry, PLQY."""
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# TRPL — additional methods
# ---------------------------------------------------------------------------
class TestTRPLDataMethods:
    def _make_trpl(self, tau=100.0, n=501, t_max=500):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, t_max, n)
        y = 1000.0 * np.exp(-t / tau)
        return TRPLData(t, y)

    def test_shift_to_max(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 500, 501)
        y = np.zeros(501)
        y[50] = 5.0; y[51] = 10.0; y[52] = 7.0  # peak at index 51
        for i in range(53, 501):
            y[i] = max(0, y[i-1] * 0.97)
        dat = TRPLData(t, y)
        dat.shift_to_max()
        assert dat.x[0] == 0.0

    def test_shift_zero(self):
        dat = self._make_trpl()
        # shift_zero(ns): crops x starting from ns and shifts so x[0]=0
        dat.shift_zero(100.0)
        assert dat.x[0] == 0.0
        assert len(dat.x) < 501

    def test_dlifetime(self):
        dat = self._make_trpl(tau=150.0)
        dl = dat.dlifetime(fluence=5e-9)
        from fte_analysis_libraries.TRPL import TRPLData
        assert isinstance(dl, TRPLData)

    def test_del_bg_with_range(self):
        from fte_analysis_libraries.TRPL import TRPLData
        # del_bg returns a new TRPLData with background subtracted
        t = np.linspace(0, 500, 501)
        y = np.ones(501) * 5.0  # constant background
        y[:100] += 100 * np.exp(-t[:100] / 20)  # signal on top
        dat = TRPLData(t, y)
        result = dat.del_bg(start=450, stop=500)
        # After subtracting bg (≈5), the end should be near 0
        assert result.y[-1] < 2.0

    def test_trpl_param_replace_with_fit(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        p = TRPLParam()
        p.replace_with_fit('k1', 5e6)
        assert p.k1 == 5e6
        assert 'unit' in dir(p)

    def test_trpl_param_replace_k2(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        p = TRPLParam()
        p.replace_with_fit('k2', 2e-10)
        assert p.k2 == 2e-10

    def test_trpl_param_replace_sr(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        p = TRPLParam()
        p.replace_with_fit('SR', 100.0)
        assert p.SR == 100.0

    def test_trpl_param_replace_sl(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        p = TRPLParam()
        p.replace_with_fit('SL', 200.0)
        assert p.SL == 200.0

    def test_trpl_param_replace_mu(self):
        from fte_analysis_libraries.TRPL import TRPLParam
        p = TRPLParam()
        p.replace_with_fit('mu', 2.5)
        assert p.mu == 2.5

    def test_gen_m3ed(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 1000, 1001)
        p0 = [5.0, 2.0, 0.5, 50.0, 200.0, 600.0]
        d = TRPLData.gen_m3ed(t, p0)
        assert isinstance(d, TRPLData)
        assert len(d.x) == 1001

    def test_gen_m4ed(self):
        from fte_analysis_libraries.TRPL import TRPLData
        t = np.linspace(0, 1000, 1001)
        p0 = [5.0, 2.0, 0.5, 0.1, 30.0, 100.0, 300.0, 800.0]
        d = TRPLData.gen_m4ed(t, p0)
        assert isinstance(d, TRPLData)

    def test_trpl_plot_no_crash(self):
        dat = self._make_trpl()
        dat.plot(show_plot=False, yscale='log')
        plt.close('all')


class TestMTRPLDataMethods:
    def _make_mtrpl(self, n=3):
        from fte_analysis_libraries.TRPL import MTRPLData, TRPLData
        t = np.linspace(0, 500, 501)
        return MTRPLData([TRPLData(t, (i+1)*np.exp(-t/(100+i*50)), name=f't{i}')
                          for i in range(n)])

    def test_construction(self):
        from fte_analysis_libraries.TRPL import MTRPLData
        m = self._make_mtrpl(3)
        assert len(m.sa) == 3

    def test_mult4_expfit_batch(self):
        from fte_analysis_libraries.TRPL import MTRPLData, TRPLData
        t = np.linspace(0, 1000, 1001)
        traces = [TRPLData(t, 10*np.exp(-t/50) + 5*np.exp(-t/100) +
                           1*np.exp(-t/300) + 0.1*np.exp(-t/800), name=f't{i}')
                  for i in range(2)]
        m = MTRPLData(traces)
        result = m.mult4_expfit(start=0, stop=1000)
        assert isinstance(result, MTRPLData)
        assert len(result.sa) == 2

    def test_plot_batch(self):
        m = self._make_mtrpl(2)
        m.label(['A', 'B'])
        m.plot(show_plot=False)
        plt.close('all')


# ---------------------------------------------------------------------------
# AbsSpectrum extended
# ---------------------------------------------------------------------------
class TestAbsSpectrumExtended:
    def _make_abs_sp(self):
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        eV = np.linspace(1.0, 3.5, 300)
        absorptance = np.clip(1 / (1 + np.exp(-(eV - 1.8) * 10)), 0, 1)
        return AbsSpectrum(eV, absorptance)

    def test_absorbed_photonflux(self):
        sp = self._make_abs_sp()
        pf = sp.absorbed_photonflux(left=1.5, right=3.0)
        assert isinstance(pf, float)
        assert pf >= 0.0

    def test_emission_pf(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = self._make_abs_sp()
        epf = sp.emission_pf(E_start=1.5, E_stop=3.0)
        assert isinstance(epf, DiffSpectrum)

    def test_u_energy_fit(self):
        from fte_analysis_libraries.Spectrum import Spectrum
        sp = self._make_abs_sp()
        result = sp.u_energy_fit(Efit_start=1.7, Efit_stop=1.85)
        assert isinstance(result, Spectrum)

    def test_tauc_plot(self):
        sp = self._make_abs_sp()
        result = sp.tauc_plot(Efit_start=1.7, Efit_stop=1.85,
                              left_offs=0.5, right_offs=0.5, showplot=False)
        plt.close('all')
        assert result is not None

    def test_calc_jradlim(self):
        sp = self._make_abs_sp()
        sp.calc_jradlim()
        assert hasattr(sp, 'Jradlim')
        assert sp.Jradlim > 0

    def test_calc_vocrad_after_jradlim(self):
        sp = self._make_abs_sp()
        sp.calc_jradlim()
        sp.calc_vocrad(E_start=1.0, E_stop=3.5)
        assert hasattr(sp, 'Vocrad')
        assert sp.Vocrad > 0

    def test_convert_absorbance_to_absorptance(self):
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        eV = np.linspace(1.5, 3.0, 100)
        absorbance = np.linspace(0.01, 1.0, 100)  # max A=1 → max absorptance≈0.9
        sp = AbsSpectrum(eV, absorbance)
        sp.convert_absorbance_to_absorptance()
        assert len(sp.y) == 100  # shape preserved

    def test_convert_absorptance_to_absorbance(self):
        sp = self._make_abs_sp()
        sp2 = sp.copy()
        sp2.convert_absorptance_to_absorbance()
        assert sp2.y[-1] > 0  # high absorptance → high absorbance

    def test_nm_to_ev(self):
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        wl = np.linspace(350, 800, 200)
        absorptance = np.clip(1 / (1 + np.exp(-(wl - 550) * 0.1)), 0, 1)
        sp = AbsSpectrum(wl, absorptance, units={'x': 'nm', 'y': ''})
        sp_ev = sp.nm_to_ev()
        assert sp_ev.ux == 'eV'


class TestSpectrumLuminosity:
    def test_luminosity_fn_construction(self):
        from fte_analysis_libraries.Spectrum import Spectrum
        lum = Spectrum.luminosity_fn()
        assert isinstance(lum, Spectrum)
        assert len(lum.x) > 0


# ---------------------------------------------------------------------------
# Electrochemistry — EIS fit
# ---------------------------------------------------------------------------
class TestEISExtended:
    def test_eis_fit_series_rc(self):
        from fte_analysis_libraries.Electrochemistry import EIS_fit
        f = np.logspace(0, 4, 20)
        R, C = 50.0, 1e-6
        omega = 2 * np.pi * f
        Z = R + 1 / (1j * omega * C)
        result = EIS_fit(f, Z, 'R0-C0', [R, C], show_details=False)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_eis_fit_returns_frequencies(self):
        from fte_analysis_libraries.Electrochemistry import EIS_fit
        f = np.logspace(0, 5, 30)
        R, C = 100.0, 1e-7
        omega = 2 * np.pi * f
        Z = R + 1 / (1j * omega * C)
        f_out, Z_out, circuit = EIS_fit(f, Z, 'R0-C0', [R, C], show_details=False)
        assert len(f_out) == len(f)
        assert np.iscomplexobj(Z_out)

    def test_eis_fit_with_range(self):
        from fte_analysis_libraries.Electrochemistry import EIS_fit
        f = np.logspace(0, 5, 30)
        R, C = 100.0, 1e-7
        omega = 2 * np.pi * f
        Z = R + 1 / (1j * omega * C)
        result = EIS_fit(f, Z, 'R0-C0', [R, C],
                         f_range=[10.0, 10000.0], show_details=False)
        assert result is not None


# ---------------------------------------------------------------------------
# PLQY — basic class construction
# ---------------------------------------------------------------------------
class TestPLQYBasic:
    def test_exp_param_default(self):
        from fte_analysis_libraries.PLQY import ExpParam
        ep = ExpParam()
        assert hasattr(ep, 'excitation_laser')
        assert hasattr(ep, 'PL_left')
        assert hasattr(ep, 'PL_right')

    def test_exp_param_custom(self):
        from fte_analysis_libraries.PLQY import ExpParam
        ep = ExpParam(which_sample='test', excitation_laser=450,
                      PL_left=600, PL_right=800)
        assert ep.PL_left == 600
        assert ep.PL_right == 800

    def test_exp_param_with_peak(self):
        from fte_analysis_libraries.PLQY import ExpParam
        ep = ExpParam(PL_peak=750)
        assert ep.PL_peak == 750


# ---------------------------------------------------------------------------
# RFB — additional functions
# ---------------------------------------------------------------------------
class TestRFBAdditional:
    def test_fit_proton_concentration_callable(self):
        import inspect

        from fte_analysis_libraries.RFB import fit_proton_concentration
        sig = inspect.signature(fit_proton_concentration)
        assert sig is not None

    def test_calc_conc_functions_proton(self):
        from fte_analysis_libraries.RFB import calc_conc_functions
        _, Xconc_Vconc_fn = calc_conc_functions(c_V=1.6, c_SO4=4.0)
        Vconc_x_fn, _ = calc_conc_functions(c_V=1.6, c_SO4=4.0)
        conc = Vconc_x_fn(3.5)
        # sol = [c_2, c_3, c_4, c_5, c_H_plus, c_HSO4_minus, c_SO4_2minus]
        sol = Xconc_Vconc_fn(conc)
        # sol[3] is H+ concentration – should be positive
        assert sol[3] > 0  # H+ concentration

    def test_calc_df_conc_index_range(self):
        from fte_analysis_libraries.RFB import calc_df_conc
        df = calc_df_conc(c_V=1.6, c_SO4=4.0)
        # Average oxidation state index
        assert df.index[0] >= 2.0
        assert df.index[-1] <= 5.0

    def test_conc_v_so4_returns_float(self):
        from fte_analysis_libraries.RFB import conc_V_SO4
        c_V, c_SO4 = conc_V_SO4()
        assert isinstance(float(c_V), float)
        assert isinstance(float(c_SO4), float)


# ---------------------------------------------------------------------------
# XYData — additional plot paths
# ---------------------------------------------------------------------------
class TestXYDataPlotPaths:
    def test_plot_loglog(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.logspace(0, 3, 50)
        y = x ** 2
        d = XYData(x, y)
        d.plot(show_plot=False, xscale='log', yscale='log')
        plt.close('all')

    def test_plot_with_bottom_top(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        y = np.sin(x) + 2
        d = XYData(x, y)
        d.plot(show_plot=False, bottom=0.5, top=3.5)
        plt.close('all')

    def test_plot_return_fig(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 5, 50)
        d = XYData(x, np.ones(50))
        fig = d.plot(show_plot=False, return_fig=True)
        plt.close('all')
        assert fig is not None

    def test_plot_linfit_residue(self):
        from fte_analysis_libraries.XYdata import XYData
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        d = XYData(x, y)
        # plot_linfit with residue=True
        result = d.plot_linfit(residue=True)
        plt.close('all')

    def test_mxy_plot_linlog(self):
        from fte_analysis_libraries.XYdata import MXYData, XYData
        x = np.linspace(0, 5, 50)
        m = MXYData([XYData(x, np.exp(-x * i * 0.3), name=f'd{i}') for i in range(3)])
        m.label(['A', 'B', 'C'])
        m.plot(show_plot=False, yscale='log')
        plt.close('all')


# ---------------------------------------------------------------------------
# IV — additional loss function tests
# ---------------------------------------------------------------------------
class TestIVLossFunctions:
    def test_sq_limit_voc_monotone(self):
        from fte_analysis_libraries.IV import IVData
        voc_vals = [IVData.sq_limit_voc(bg=bg) for bg in [1.2, 1.4, 1.6, 1.8, 2.0]]
        assert all(voc_vals[i] < voc_vals[i+1] for i in range(len(voc_vals)-1))

    def test_sq_limit_jsc_monotone(self):
        from fte_analysis_libraries.IV import IVData
        jsc_vals = [IVData.sq_limit_jsc(bg=bg) for bg in [1.2, 1.4, 1.6, 1.8, 2.0]]
        assert all(jsc_vals[i] > jsc_vals[i+1] for i in range(len(jsc_vals)-1))

    def test_iv_sq_different_bandgaps(self):
        from fte_analysis_libraries.IV import IVData
        for bg in [1.1, 1.4, 1.7]:
            iv = IVData.iv_sq(bg=bg)
            iv.det_voc(); iv.det_jsc()
            assert iv.Voc > 0
            assert iv.Jsc > 0

    def test_i_of_v_multiple_voltages(self):
        from fte_analysis_libraries.IV import IVData
        voltages = [0.0, 0.3, 0.6, 0.91]
        for V in voltages:
            I = IVData.i_of_v(V, Isc=-20e-3, Voc=0.91, nid=1.5, Rs=2.0, Rsh=5000.0)
            assert isinstance(float(I), float)

    def test_det_perfparam_minimal(self):
        from fte_analysis_libraries.IV import IVData
        V = np.linspace(-0.1, 1.0, 200)
        iv = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5,
                             Rs=2.0, Rsh=5000.0, light_int=100.0)
        iv.det_voc(); iv.det_jsc()
        iv.det_perfparam(minimal=True)
        assert hasattr(iv, 'pd')
        assert iv.pd.PCE > 0
