"""Tests for TRPL.py — TRPLData fitting."""
import numpy as np
import pytest

from fte_analysis_libraries.TRPL import TRPLData, initial_carrier_conc, one_sun_carrier_conc


class TestTRPLDataConstruction:
    def test_basic(self):
        ns = np.linspace(0, 500, 501)
        cts = np.exp(-ns / 100.0)
        t = TRPLData(ns, cts)
        assert len(t.x) == 501
        assert t.x[0] == 0.0

    def test_copy(self):
        ns = np.linspace(0, 200, 201)
        cts = np.exp(-ns / 50.0)
        t = TRPLData(ns, cts, name='orig')
        c = t.copy()
        c.x[0] = -999.0
        assert t.x[0] == 0.0


class TestMonoExpFit:
    def _make_decay(self, a=1.0, tau=150.0, n_pts=501, t_max=500.0):
        """Synthetic mono-exponential decay."""
        ns = np.linspace(0, t_max, n_pts)
        cts = a * np.exp(-ns / tau)
        return TRPLData(ns, cts)

    def test_tau_recovery_within_1pct(self):
        true_tau = 150.0
        t = self._make_decay(a=1.0, tau=true_tau)
        fit = t.mono_expfit(start=0.0, stop=500.0, p0=(1.0, 100.0))
        recovered_tau = fit.popt[1]
        assert abs(recovered_tau - true_tau) / true_tau < 0.01

    def test_amplitude_recovery_within_1pct(self):
        true_a = 2.5
        t = self._make_decay(a=true_a, tau=200.0)
        fit = t.mono_expfit(start=0.0, stop=500.0, p0=(2.0, 150.0))
        recovered_a = fit.popt[0]
        assert abs(recovered_a - true_a) / true_a < 0.01

    def test_fit_returns_trpl_data(self):
        t = self._make_decay()
        fit = t.mono_expfit(start=0.0, stop=500.0)
        assert isinstance(fit, TRPLData)
        assert len(fit.x) == len(t.x)

    def test_fit_has_popt(self):
        t = self._make_decay()
        fit = t.mono_expfit(start=0.0)
        assert hasattr(fit, 'popt')
        assert len(fit.popt) == 2


class TestMult2ExpFit:
    def test_mult2_fit(self):
        ns = np.linspace(0, 500, 501)
        # bi-exponential with tau1=50, tau2=200
        cts = 0.6 * np.exp(-ns / 50.0) + 0.4 * np.exp(-ns / 200.0)
        t = TRPLData(ns, cts)
        fit = t.mult2_expfit(start=0, stop=500, p0=(0.6, 0.4, 50, 200))
        assert hasattr(fit, 'popt')
        assert len(fit.popt) == 4


class TestHelperFunctions:
    def test_initial_carrier_conc(self):
        # wavelength 532 nm, 500 nm film, 1 μJ/cm² fluence
        n0 = initial_carrier_conc(wavelength=532, film_thickness=500, fluence=1e-6)
        assert n0 > 0

    def test_one_sun_carrier_conc(self):
        cc = one_sun_carrier_conc(lifetime=1000, bg=1.55, film_thickness=500)
        assert cc > 0
