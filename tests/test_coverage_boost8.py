"""Eighth coverage-boost: Spectrum.py DiffSpectrum calc_*, PELSpectrum, Spectra I/O."""
import warnings
import tempfile
import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# DiffSpectrum.calc_illuminance — multiple uy branches
# ---------------------------------------------------------------------------
class TestCalcIlluminance:
    def test_calc_illuminance_irradiance(self):
        """uy = 'W/[m2 nm]' branch (line 1070)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        ill = am.calc_illuminance()
        assert ill > 0

    def test_calc_illuminance_photonflux(self):
        """uy = '1/[s m2 nm]' branch (line 1072-1075)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral photon flux')
        ill = am.calc_illuminance()
        assert ill > 0

    def test_calc_illuminance_illuminance_unit(self):
        """uy = 'lm/[m2 nm]' branch (line 1076-1077)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        ill_sp = am.irradiance_to_illuminance()
        ill = ill_sp.calc_illuminance()
        assert ill > 0

    def test_calc_illuminance_wrong_unit(self):
        """else branch (line 1080-1081)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(400, 720, 100)
        sp = DiffSpectrum(wl, np.ones(100),
                          units={'x': 'nm', 'y': 'WRONG_UNIT'})
        result = sp.calc_illuminance()
        assert result is None  # prints warning, returns nothing

    def test_calc_illuminance_wrong_x_unit(self):
        """ux != 'nm' branch (line 1083)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        eV = np.linspace(1.5, 3.0, 100)
        sp = DiffSpectrum(eV, np.ones(100),
                          units={'x': 'eV', 'y': 'W/[m2 eV]'})
        result = sp.calc_illuminance()
        assert result is None


# ---------------------------------------------------------------------------
# DiffSpectrum.normalize_to_lux (lines 1091-1092)
# ---------------------------------------------------------------------------
class TestNormalizeToLux:
    def test_normalize_to_lux(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=400, right=720, delta=1.0,
                                   y_unit='Spectral irradiance')
        am.normalize_to_lux(1000.0)
        # After normalization, calc_illuminance should equal 1000 lx
        ill = am.calc_illuminance()
        assert abs(ill - 1000.0) < 10.0


# ---------------------------------------------------------------------------
# DiffSpectrum.calc_irradiance (lines 1106-1111)
# ---------------------------------------------------------------------------
class TestCalcIrradiance:
    def test_calc_irradiance_nm(self):
        """ux='nm', uy='W/[m2 nm]' branch (line 1107)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=300, right=2500, delta=1.0,
                                   y_unit='Spectral irradiance')
        irr = am.calc_irradiance()
        assert 90 < irr < 110  # AM1.5G ≈ 100 mW/cm²

    def test_calc_irradiance_wrong_unit(self):
        """else branch (lines 1109-1111)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        wl = np.linspace(300, 800, 200)
        sp = DiffSpectrum(wl, np.ones(200),
                          units={'x': 'nm', 'y': '1/[s m2 nm]'})
        result = sp.calc_irradiance()
        assert result is None


# ---------------------------------------------------------------------------
# DiffSpectrum.bb_spectrum (lines 1161-1168)
# ---------------------------------------------------------------------------
class TestBBSpectrum:
    def test_bb_spectrum(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am_ev = DiffSpectrum.am15_ev(left=1.0, right=4.0, delta=0.01)
        bb = DiffSpectrum.bb_spectrum(am_ev, T=300, fit_eV=2.0)
        assert bb is not None
        assert bb.ux == 'eV'


# ---------------------------------------------------------------------------
# DiffSpectrum.integrated_current (lines 1193-1204)
# ---------------------------------------------------------------------------
class TestIntegratedCurrent:
    def test_integrated_current_no_eqe(self):
        """EQE=None branch (line 1195)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=300, right=1200, delta=2.0)
        ic = am.integrated_current(EQE=None)
        assert ic is not None
        assert ic.y[-1] > 0  # total integrated current at end

    def test_integrated_current_with_eqe(self):
        """EQE != None branch (line 1197)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum, EQESpectrum
        am = DiffSpectrum.am15_nm(left=300, right=1200, delta=2.0)
        eqe = EQESpectrum.eqe100(Eg=1.2)
        ic = am.integrated_current(EQE=eqe)
        assert ic is not None


# ---------------------------------------------------------------------------
# DiffSpectrum.integrated_irradiance (lines 1223-1224)
# ---------------------------------------------------------------------------
class TestIntegratedIrradiance:
    def test_integrated_irradiance(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=300, right=1200, delta=1.0,
                                   y_unit='Spectral irradiance')
        ii = am.integrated_irradiance()
        assert ii is not None
        assert ii.y[-1] > 0


# ---------------------------------------------------------------------------
# DiffSpectrum.calculate_irradiance_illuminance (lines 1250-1258)
# ---------------------------------------------------------------------------
class TestCalculateIrradianceIlluminance:
    def test_calculate_with_show(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=300, right=2500, delta=1.0)
        irr, ill = am.calculate_irradiance_illuminance(left=400, right=720, show=True)
        assert irr > 0
        assert ill > 0

    def test_calculate_no_show(self):
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        am = DiffSpectrum.am15_nm(left=300, right=2500, delta=1.0)
        irr, ill = am.calculate_irradiance_illuminance(left=400, right=720, show=False)
        assert irr > 0


# ---------------------------------------------------------------------------
# PELSpectrum.absorptance (lines 1299-1305)
# ---------------------------------------------------------------------------
class TestPELSpectrumAbsorptance:
    def _make_pel(self, E):
        from fte_analysis_libraries.Spectrum import PELSpectrum, DiffSpectrum
        from fte_analysis_libraries.General import k, T_RT, q
        A = 1e20
        y = A * E**2 * np.exp(-(E * q) / (k * T_RT))
        return PELSpectrum(E, y,
                           quants={'x': 'Photon energy', 'y': 'PL'},
                           units={'x': 'eV', 'y': 'counts'})

    def test_absorptance_matching_x(self):
        """self.x == bb.x → compute absorptance (line 1303-1304)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        E = np.linspace(1.8, 3.0, 200)
        pel = self._make_pel(E)
        bb = DiffSpectrum.bb_spectrum(pel, T=300, fit_eV=2.0)
        ab = pel.absorptance(bb, eV=2.0, a_eV=0.9)
        assert ab is not None

    def test_absorptance_mismatched_x(self):
        """self.x != bb.x → returns self (line 1300-1301)."""
        from fte_analysis_libraries.Spectrum import DiffSpectrum
        E1 = np.linspace(1.8, 3.0, 200)
        E2 = np.linspace(1.9, 3.1, 200)  # different x
        pel = self._make_pel(E1)
        bb = DiffSpectrum(E2, np.ones(200) * 1e10,
                          quants={'x': 'Photon energy', 'y': 'BB'},
                          units={'x': 'eV', 'y': ''})
        ab = pel.absorptance(bb, eV=2.0, a_eV=0.9)
        # returns self (the PELSpectrum) unchanged
        assert np.allclose(ab.y, pel.y)


# ---------------------------------------------------------------------------
# Spectra.load_multiple (lines 1502-1529)
# ---------------------------------------------------------------------------
class TestSpectraLoadMultiple:
    def test_load_multiple(self):
        import pandas as pd
        from fte_analysis_libraries.Spectrum import Spectra
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(300, 800, 101)
            df = pd.DataFrame({
                'Wavelength (nm)': wl,
                'PL 1 (counts)': np.exp(-(wl - 600)**2 / 100),
                'PL 2 (counts)': np.exp(-(wl - 620)**2 / 100),
            })
            fp = os.path.join(tmp, 'spectra.csv')
            df.to_csv(fp, index=False)
            sa = Spectra.load_multiple(tmp, 'spectra.csv')
            assert len(sa.sa) == 2

    def test_load_multiple_with_quants(self):
        import pandas as pd
        from fte_analysis_libraries.Spectrum import Spectra
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(300, 800, 50)
            df = pd.DataFrame({
                'Wavelength (nm)': wl,
                'Intensity (counts)': np.ones(50),
            })
            fp = os.path.join(tmp, 'sp.csv')
            df.to_csv(fp, index=False)
            sa = Spectra.load_multiple(tmp, 'sp.csv', take_quants_and_units_from_file=True)
            assert sa.sa[0].qx == 'Wavelength'
            assert sa.sa[0].ux == 'nm'


# ---------------------------------------------------------------------------
# Spectra.load_individual (lines 1537-1574) — multiple class names
# ---------------------------------------------------------------------------
class TestSpectraLoadIndividual:
    def _make_temp_csvs(self, tmp, n=2):
        import pandas as pd
        for i in range(n):
            wl = np.linspace(400, 700, 50)
            df = pd.DataFrame({'Wavelength (nm)': wl, 'PL (cts)': np.ones(50) * (i+1)})
            df.to_csv(os.path.join(tmp, f'spec_{i}.csv'), index=False)

    def test_load_individual_spectra(self):
        from fte_analysis_libraries.Spectrum import Spectra
        with tempfile.TemporaryDirectory() as tmp:
            self._make_temp_csvs(tmp)
            sa = Spectra.load_individual(tmp)
            assert len(sa.sa) == 2

    def test_load_individual_spectra_with_quants(self):
        from fte_analysis_libraries.Spectrum import Spectra
        with tempfile.TemporaryDirectory() as tmp:
            self._make_temp_csvs(tmp)
            sa = Spectra.load_individual(tmp, take_quants_and_units_from_file=True)
            assert sa.sa[0].qx == 'Wavelength'
            assert sa.sa[0].ux == 'nm'

    def test_load_individual_absspectra(self):
        import pandas as pd
        from fte_analysis_libraries.Spectrum import AbsSpectra
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(400, 700, 50)
            df = pd.DataFrame({'Wavelength': wl, 'Absorptance': np.ones(50) * 0.5})
            df.to_csv(os.path.join(tmp, 'ab.csv'), index=False)
            sa = AbsSpectra.load_individual(tmp)
            assert len(sa.sa) == 1

    def test_load_individual_diffspectra(self):
        import pandas as pd
        from fte_analysis_libraries.Spectrum import DiffSpectra
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(400, 700, 50)
            df = pd.DataFrame({'Wavelength': wl, 'PF': np.ones(50)})
            df.to_csv(os.path.join(tmp, 'df.csv'), index=False)
            sa = DiffSpectra.load_individual(tmp)
            assert len(sa.sa) == 1

    def test_load_individual_pelspectra(self):
        import pandas as pd
        from fte_analysis_libraries.Spectrum import PELSpectra
        with tempfile.TemporaryDirectory() as tmp:
            wl = np.linspace(400, 700, 50)
            df = pd.DataFrame({'Wavelength': wl, 'PL': np.ones(50)})
            df.to_csv(os.path.join(tmp, 'pl.csv'), index=False)
            sa = PELSpectra.load_individual(tmp)
            assert len(sa.sa) == 1
