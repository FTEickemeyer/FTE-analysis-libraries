"""Tests for Spectrum.py."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fte_analysis_libraries.General import f1240, q
from fte_analysis_libraries.Spectrum import (
    AbsSpectrum,
    DiffSpectrum,
    EQESpectrum,
    Spectra,
    Spectrum,
    above_bg_photon_flux,
)


class TestSpectrumConstruction:
    def test_basic(self):
        wl = np.linspace(400, 800, 100)
        counts = np.ones(100)
        sp = Spectrum(wl, counts)
        assert len(sp.x) == 100

    def test_with_metadata(self):
        wl = np.linspace(300, 900, 200)
        y = np.ones(200)
        sp = Spectrum(wl, y, quants={'x': 'Wavelength', 'y': 'Intensity'},
                      units={'x': 'nm', 'y': 'counts/s'}, name='test_spectrum')
        assert sp.qx == 'Wavelength'
        assert sp.name == 'test_spectrum'

    def test_copy(self):
        wl = np.linspace(400, 700, 50)
        y = np.linspace(1, 50, 50)
        sp = Spectrum(wl, y, name='orig')
        c = sp.copy()
        c.x[0] = 0.0
        assert sp.x[0] != 0.0

    def test_normalize(self):
        wl = np.linspace(400, 700, 100)
        y = np.linspace(1, 100, 100)
        sp = Spectrum(wl, y)
        sp.normalize()
        assert abs(max(sp.y) - 1.0) < 1e-10


class TestEQESpectrum:
    def _make_eqe(self, wl_max=700):
        wl = np.linspace(300, 800, 200)
        eqe = np.where(wl < wl_max, 80.0, 0.0)
        return EQESpectrum(wl, eqe,
                           quants={'x': 'Wavelength', 'y': 'EQE'},
                           units={'x': 'nm', 'y': '%'})

    def test_construction(self):
        eqe = self._make_eqe()
        assert len(eqe.x) == 200

    def test_jsc_positive(self):
        eqe = self._make_eqe()
        jsc = eqe.calc_jsc()
        assert jsc > 0

    def test_jsc_plausible(self):
        eqe = self._make_eqe()
        jsc = eqe.calc_jsc()
        assert 5.0 < jsc < 40.0

    def test_eqe100_constructor(self):
        eqe = EQESpectrum.eqe100(Eg=1.6)
        assert len(eqe.x) > 0
        assert eqe.ux == 'nm'
        assert eqe.uy == '%'

    def test_eqe100_jsc(self):
        eqe = EQESpectrum.eqe100(Eg=1.6)
        jsc = eqe.calc_jsc()
        assert jsc > 0

    def test_jsc_eV_units(self):
        eV = np.linspace(1.0, 4.0, 300)
        eqe = np.where(eV < 3.0, 80.0, 0.0)
        sp = EQESpectrum(eV, eqe,
                         quants={'x': 'Photon energy', 'y': 'EQE'},
                         units={'x': 'eV', 'y': '%'})
        jsc = sp.calc_jsc()
        assert jsc > 0


class TestDiffSpectrum:
    def test_construction(self):
        eV = np.linspace(1.0, 4.0, 100)
        pf = np.exp(-((eV - 2.5) ** 2) / 0.1)
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'Photon energy', 'y': 'Photon flux'},
                          units={'x': 'eV', 'y': '1/(s cm2 eV)'})
        assert len(sp.x) == 100

    def test_photonflux_positive(self):
        eV = np.linspace(1.0, 4.0, 200)
        pf = np.abs(np.sin(eV)) + 0.01
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'Photon energy', 'y': 'Spectral photon flux'},
                          units={'x': 'eV', 'y': '1/(s cm2 eV)'})
        flux = sp.photonflux()
        assert flux > 0

    def test_am15_nm_loads(self):
        am = DiffSpectrum.am15_nm(left=300, right=1200, delta=1.0)
        assert len(am.x) > 0
        assert am.ux == 'nm'

    def test_am15_ev_loads(self):
        am = DiffSpectrum.am15_ev(left=0.5, right=4.0, delta=0.01)
        assert len(am.x) > 0
        assert am.ux == 'eV'

    def test_photonflux_range(self):
        eV = np.linspace(1.0, 4.0, 200)
        pf = np.ones(200)
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'E', 'y': 'PF'},
                          units={'x': 'eV', 'y': '1/(s cm2 eV)'})
        flux = sp.photonflux(start=1.5, stop=3.5)
        assert flux > 0

    def test_calc_integrated_photonflux(self):
        eV = np.linspace(1.0, 4.0, 200)
        pf = np.ones(200) * 2.0
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'E', 'y': 'PF'},
                          units={'x': 'eV', 'y': '1/(s cm2 eV)'})
        flux = sp.calc_integrated_photonflux(start=1.0, stop=4.0)
        assert abs(flux - 6.0) < 0.1  # 2.0 * 3.0 eV range

    def test_nm_to_ev(self):
        wl = np.linspace(300, 800, 100)
        sp = DiffSpectrum(wl, np.ones(100),
                          quants={'x': 'Wavelength', 'y': 'PF'},
                          units={'x': 'nm', 'y': '1/[s m2 nm]'})
        ev_sp = sp.nm_to_ev()
        assert ev_sp.ux == 'eV'
        assert min(ev_sp.x) > 0


class TestAbsSpectrum:
    def test_construction(self):
        eV = np.linspace(1.0, 3.5, 100)
        y = 1 / (1 + np.exp(-(eV - 1.8) * 10))
        sp = AbsSpectrum(eV, y)
        assert len(sp.x) == 100
        assert sp.qy == 'Absorptance'
        assert sp.ux == 'eV'


class TestSpectra:
    def test_construction(self):
        items = []
        for i in range(3):
            wl = np.linspace(400, 700, 50)
            sp = Spectrum(wl, np.ones(50) * float(i + 1), name=f'sp_{i}')
            items.append(sp)
        sa = Spectra(items)
        assert sa.n_y == 3

    def test_plot_no_crash(self):
        items = [Spectrum(np.linspace(400, 700, 50), np.ones(50), name=f'sp_{i}')
                 for i in range(3)]
        sa = Spectra(items)
        sa.label([f'label_{i}' for i in range(3)])
        sa.plot(show_plot=False)
        plt.close('all')


class TestAboveBgPhotonFlux:
    def test_returns_positive(self):
        result = above_bg_photon_flux(1.6)
        assert result > 0

    def test_higher_bg_lower_flux(self):
        flux_low = above_bg_photon_flux(1.2)
        flux_high = above_bg_photon_flux(2.0)
        assert flux_low > flux_high

    def test_reasonable_magnitude(self):
        # AM1.5G above 1.1 eV — result is in photons/s/m², ~2.7e21
        result = above_bg_photon_flux(1.1)
        assert 1e20 < result < 1e23
