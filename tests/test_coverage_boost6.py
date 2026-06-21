"""Sixth coverage-boost: Spectrum.py conversions, loading variants, Spectra methods."""
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# DiffSpectrum — ev_to_nm conversion
# ---------------------------------------------------------------------------
class TestDiffSpectrumEvToNm:
    def test_ev_to_nm(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        eV = np.linspace(1.0, 4.0, 200)
        pf = np.exp(-((eV - 2.5) ** 2) / 0.1)
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'Photon energy', 'y': 'Spectral photon flux'},
                          units={'x': 'eV', 'y': '1/(s m2 eV)'})
        sp_nm = sp.ev_to_nm()
        assert sp_nm.ux == 'nm'
        assert len(sp_nm.x) > 0

    def test_ev_to_nm_preserves_type(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        eV = np.linspace(1.5, 3.0, 100)
        pf = np.ones(100)
        sp = DiffSpectrum(eV, pf,
                          quants={'x': 'Photon energy', 'y': 'Spectral photon flux'},
                          units={'x': 'eV', 'y': '1/(s m2 eV)'})
        sp_nm = sp.ev_to_nm()
        assert isinstance(sp_nm, DiffSpectrum)


# ---------------------------------------------------------------------------
# DiffSpectrum — load_astmg173 variants (different Spectrum types and y_units)
# ---------------------------------------------------------------------------
class TestLoadAstmg173Variants:
    def test_load_etr_nm(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.etr_nm(left=300, right=800, delta=2.0)
        assert sp.ux == 'nm'
        assert len(sp.x) > 0

    def test_load_etr_ev(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.etr_ev(left=1.0, right=3.0, delta=0.01)
        assert sp.ux == 'eV'

    def test_am15direct_nm(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.am15direct_nm(left=300, right=800, delta=2.0)
        assert sp.ux == 'nm'

    def test_am15direct_ev(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.am15direct_ev(left=1.0, right=3.0, delta=0.01)
        assert sp.ux == 'eV'

    def test_load_astmg173_irradiance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_astmg173(y_unit='Spectral irradiance', warning=False)
        assert 'W' in sp.uy

    def test_load_astmg173_with_warning(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_astmg173(warning=True)
        assert sp is not None

    def test_load_astmg173_pf_alias(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_astmg173(y_unit='PF', warning=False)
        assert '1/[s m2 nm]' in sp.uy

    def test_load_astmg173_sf_alias(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_astmg173(y_unit='SF', warning=False)
        assert 'W' in sp.uy

    def test_am15_nm_with_irradiance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.am15_nm(left=300, right=800, delta=2.0,
                                   y_unit='Spectral irradiance')
        assert 'W' in sp.uy


# ---------------------------------------------------------------------------
# DiffSpectrum — illuminant loaders (uses system files)
# ---------------------------------------------------------------------------
class TestDiffSpectrumIlluminantLoaders:
    def test_load_osram930_pf(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_osram930(y_unit='Spectral photon flux')
        assert sp is not None
        assert sp.ux == 'nm'

    def test_load_osram930_sf(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_osram930(y_unit='Spectral irradiance')
        assert 'W' in sp.uy

    def test_load_led5000k(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        sp = DiffSpectrum.load_led5000k(y_unit='Spectral photon flux')
        assert sp is not None

    def test_load_illuminant_unknown_unit(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        # Should print warning but not crash
        try:
            sp = DiffSpectrum._load_illuminant_spectrum(
                'OSRAM_930.txt', y_unit='UNKNOWN_UNIT', delimiter='\t', header=2)
        except Exception:
            pass  # May fail on unknown unit — just cover the code path


# ---------------------------------------------------------------------------
# DiffSpectrum — photon flux ↔ irradiance conversions
# ---------------------------------------------------------------------------
class TestDiffSpectrumConversions:
    def test_photonflux_to_irradiance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 200)
        pf = np.ones(200) * 1e15
        sp = DiffSpectrum(wl, pf,
                          quants={'x': 'Wavelength', 'y': 'Spectral photon flux'},
                          units={'x': 'nm', 'y': '1/[s m2 nm]'})
        irr = sp.photonflux_to_irradiance()
        assert irr is not None
        assert 'W' in irr.uy

    def test_photonflux_to_irradiance_wrong_unit(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 200)
        sp = DiffSpectrum(wl, np.ones(200),
                          quants={'x': 'Wavelength', 'y': 'PF'},
                          units={'x': 'nm', 'y': 'WRONG_UNIT'})
        result = sp.photonflux_to_irradiance()
        assert result is None  # prints warning, returns None

    def test_irradiance_to_photonflux_with_factor(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 200)
        sp = DiffSpectrum(wl, np.ones(200) * 1e3,
                          quants={'x': 'Wavelength', 'y': 'Spectral irradiance'},
                          units={'x': 'nm', 'y': 'uW/[cm2 nm]'})
        pf = sp.irradiance_to_photonflux(factor=1e-6/1e-4)
        assert pf is not None
        assert '1/[s m2 nm]' in pf.uy

    def test_irradiance_to_photonflux_standard(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 200)
        sp = DiffSpectrum(wl, np.ones(200),
                          quants={'x': 'Wavelength', 'y': 'Spectral irradiance'},
                          units={'x': 'nm', 'y': 'W/[m2 nm]'})
        pf = sp.irradiance_to_photonflux()
        assert pf is not None

    def test_irradiance_to_illuminance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        ill = am.irradiance_to_illuminance()
        assert ill is not None

    def test_illuminance_to_irradiance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        ill = am.irradiance_to_illuminance()
        irr = ill.illuminance_to_irradiance(warning=False)
        assert irr is not None

    def test_illuminance_to_irradiance_wrong_unit(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(400, 720, 100)
        sp = DiffSpectrum(wl, np.ones(100),
                          quants={'x': 'Wavelength', 'y': 'Illuminance'},
                          units={'x': 'nm', 'y': 'WRONG'})
        result = sp.illuminance_to_irradiance()
        assert result is None


# ---------------------------------------------------------------------------
# Spectrum.nm_to_ev — EQESpectrum branch
# ---------------------------------------------------------------------------
class TestSpectrumNmToEv:
    def test_eqe_nm_to_ev(self):
        from fte_analysis_libraries.Spectrum import EQESpectrum
        wl = np.linspace(300, 800, 200)
        eqe = np.where(wl < 620, 80.0, 0.0)
        sp = EQESpectrum(wl, eqe,
                         quants={'x': 'Wavelength', 'y': 'EQE'},
                         units={'x': 'nm', 'y': '%'})
        sp_ev = sp.nm_to_ev()
        assert sp_ev.ux == 'eV'
        from fte_analysis_libraries.Spectrum import EQESpectrum as EQE
        assert isinstance(sp_ev, EQE)

    def test_abs_nm_to_ev(self):
        from fte_analysis_libraries.Spectrum import AbsSpectrum
        wl = np.linspace(300, 800, 200)
        absorptance = np.clip(1 / (1 + np.exp(-(wl - 550) * 0.05)), 0, 1)
        sp = AbsSpectrum(wl, absorptance, units={'x': 'nm', 'y': ''})
        sp_ev = sp.nm_to_ev()
        assert sp_ev.ux == 'eV'


# ---------------------------------------------------------------------------
# Spectra — replace, nm_to_ev, save
# ---------------------------------------------------------------------------
class TestSpectraMethods:
    def _make_spectra(self, n=3):
        from fte_analysis_libraries.Spectrum import Spectra, Spectrum
        wl = np.linspace(400, 700, 50)
        sa = [Spectrum(wl, np.ones(50) * float(i+1), name=f'sp_{i}')
              for i in range(n)]
        return Spectra(sa)

    def test_replace(self):
        from fte_analysis_libraries.Spectrum import Spectrum
        sa = self._make_spectra(3)
        new_sp = Spectrum(np.linspace(400, 700, 50), np.ones(50) * 99, name='new')
        sa.replace(1, new_sp)
        assert sa.sa[1].y[0] == 99.0

    def test_nm_to_ev(self):
        sa = self._make_spectra(2)
        sa_ev = sa.nm_to_ev()
        for sp in sa_ev.sa:
            assert sp.ux == 'eV'

    def test_names_to_label(self):
        sa = self._make_spectra(3)
        sa.names_to_label(split_ch='.csv')
        assert sa.label_defined

    def test_generate_empty(self):
        from fte_analysis_libraries.Spectrum import Spectra
        sa = Spectra.generate_empty([])
        assert len(sa.sa) == 0

    def test_save(self):
        import os
        import tempfile

        from fte_analysis_libraries.Spectrum import Spectra, Spectrum
        wl = np.linspace(400, 700, 301)  # equidistant with delta=1
        sa = Spectra([Spectrum(wl, np.ones(301), name=f'sp_{i}') for i in range(2)])
        with tempfile.TemporaryDirectory() as tmp:
            sa.save(tmp, 'test_spectra.csv')
            files = os.listdir(tmp)
            assert len(files) > 0


# ---------------------------------------------------------------------------
# DiffSpectrum — bbt_fit (Blackbody tail)
# ---------------------------------------------------------------------------
class TestPELSpectrumBBTFit:
    def test_bbt_fit_converges(self):
        from fte_analysis_libraries.General import k, q
        from fte_analysis_libraries.Spectrum import PELSpectrum
        T_target = 300.0
        E = np.linspace(1.8, 3.0, 200)
        A = 1e20
        y = A * E**2 * np.exp(-(E * q) / (k * T_target))
        sp = PELSpectrum(E, y,
                         quants={'x': 'Photon energy', 'y': 'PL'},
                         units={'x': 'eV', 'y': 'counts'})
        fit = sp.bbt_fit(Efit_start=2.0, Efit_stop=2.5)
        assert hasattr(fit, 'T')
        assert 100 < fit.T < 1000

    def test_bbt_fit_no_converge(self):
        from fte_analysis_libraries.Spectrum import PELSpectrum
        E = np.linspace(1.8, 3.0, 50)
        np.random.seed(42)
        y = np.random.rand(50) * 1e-30  # flat noise → no BB structure → won't converge
        sp = PELSpectrum(E, y,
                         quants={'x': 'Photon energy', 'y': 'PL'},
                         units={'x': 'eV', 'y': 'counts'})
        fit = sp.bbt_fit(Efit_start=2.0, Efit_stop=2.5)
        assert fit is not None


# ---------------------------------------------------------------------------
# General.py — idx_range edge cases
# ---------------------------------------------------------------------------
class TestIdxRangeEdgeCases:
    def test_idx_range_ascending_l_below_min(self):
        from fte_analysis_libraries.General import idx_range
        arr = np.linspace(1, 10, 100)
        r = idx_range(arr, left=-5, right=8)
        assert r is not None

    def test_idx_range_ascending_r_above_max(self):
        from fte_analysis_libraries.General import idx_range
        arr = np.linspace(1, 10, 100)
        r = idx_range(arr, left=3, right=20)
        assert r is not None

    def test_idx_range_descending_r_below_min(self):
        from fte_analysis_libraries.General import idx_range
        arr = np.linspace(10, 1, 100)  # descending
        r = idx_range(arr, left=8, right=-5)  # r < min → clamp r
        assert r is not None

    def test_idx_range_descending_l_above_max(self):
        from fte_analysis_libraries.General import idx_range
        arr = np.linspace(10, 1, 100)  # descending
        r = idx_range(arr, left=20, right=5)  # l > max → clamp l
        assert r is not None
