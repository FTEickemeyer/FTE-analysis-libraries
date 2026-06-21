"""Tests for IV.py — IVData, FiveParam, PerfData."""
import numpy as np
import pytest
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fte_analysis_libraries.IV import IVData, FiveParam, PerfData
from fte_analysis_libraries.General import q, k, T_RT

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def _make_synthetic_jv(Voc=0.7, Jsc=20.0, nid=1.5, Rs=1e-9, Rsh=1e6, n_pts=201):
    """Generate a synthetic JV curve using the Lambert-W model."""
    V = np.linspace(-0.1, Voc * 1.05, n_pts)
    J0 = Jsc * np.exp(-q * Voc / (nid * k * T_RT))
    Jph = Jsc
    IV = IVData.from_j0(V, J0, Jph, nid, Rs, Rsh, light_int=100.0, name='synthetic')
    return IV


class TestFiveParam:
    def test_construction_defaults(self):
        fp = FiveParam()
        assert fp.Voc == 1.0
        assert fp.Jsc == 20.0

    def test_construction_custom(self):
        fp = FiveParam(cell_area=0.1, Voc=1.0, Jsc=20.0, nid=1.5, Rs=1e-9, Rsh=1e6)
        assert fp.Voc == 1.0
        assert fp.Jsc == 20.0

    def test_copy(self):
        fp = FiveParam(Voc=0.8, Jsc=15.0)
        c = fp.copy()
        c.Voc = 99.0
        assert fp.Voc == 0.8


class TestPerfData:
    def test_construction(self):
        pd_ = PerfData(Voc=0.7, Jsc=20.0, FF=75.0, PCE=10.5)
        assert pd_.Voc == 0.7
        assert pd_.FF == 75.0

    def test_copy(self):
        pd_ = PerfData(Voc=0.7, Jsc=20.0, FF=75.0)
        c = pd_.copy()
        c.Voc = 999.0
        assert pd_.Voc == 0.7

    def test_voc_text(self):
        pd_ = PerfData(Voc=0.712, Jsc=20.1, FF=76.5, PCE=10.9,
                       Vmpp=0.6, Jmpp=18.0, Pmpp=10.8, nid=1.4,
                       Rs=1e-3, Rsh=1e3)
        assert '0.712' in pd_.voc_text()

    def test_jsc_text(self):
        pd_ = PerfData(Jsc=20.1)
        assert '20.10' in pd_.jsc_text()

    def test_ff_text(self):
        pd_ = PerfData(FF=76.5)
        assert '76.5' in pd_.ff_text()

    def test_pce_text(self):
        pd_ = PerfData(PCE=12.3)
        assert '12.3' in pd_.pce_text()

    def test_vmpp_text(self):
        pd_ = PerfData(Vmpp=0.612)
        assert '0.612' in pd_.vmpp_text()

    def test_pmpp_text(self):
        pd_ = PerfData(Pmpp=9.87)
        assert '9.87' in pd_.pmpp_text()

    def test_nid_text(self):
        pd_ = PerfData(nid=1.45)
        assert '1.45' in pd_.nid_text()

    def test_rs_text(self):
        pd_ = PerfData(Rs=1e-3)
        text = pd_.rs_text()
        assert 'R' in text

    def test_jsc_text_ua(self):
        pd_ = PerfData(Jsc=0.02)
        text = pd_.jsc_text(uA=True)
        assert 'mu' in text or 'A' in text

    def test_sq_limit_returns_perfdata(self):
        pd_ = PerfData.sq_limit(bg=1.6, show=False)
        assert pd_.Voc is not None
        assert pd_.Jsc is not None
        assert pd_.Voc > 0
        assert pd_.Jsc > 0


class TestIVDataConstruction:
    def test_basic(self):
        V = np.linspace(-0.1, 0.8, 100)
        J = 20.0 * np.ones(100)
        iv = IVData(V, J, cell_area=1.0, light_int=100.0)
        assert len(iv.x) == 100
        assert iv.cell_area == 1.0

    def test_copy(self):
        V = np.linspace(0, 0.8, 50)
        J = np.ones(50) * 20.0
        iv = IVData(V, J, name='orig')
        c = iv.copy()
        c.x[0] = 999.0
        assert iv.x[0] != 999.0

    def test_convert_mA_to_uA(self):
        V = np.linspace(0, 0.8, 50)
        J = np.ones(50) * 20.0
        iv = IVData(V, J.copy())
        iv.convert_from_mA_to_uA()
        assert iv.uy == 'uA/cm2'
        assert abs(iv.y[0] - 20000.0) < 1e-6


class TestIVDataFromJ0:
    def test_construction(self):
        V = np.linspace(-0.1, 0.8, 100)
        iv = IVData.from_j0(V, J0=1e-10, Jph=20.0, nid=1.5, Rs=1e-6, Rsh=1e6)
        assert len(iv.x) == 100


class TestIVDataFromFp:
    def test_basic_from_fp(self):
        fp = FiveParam(Voc=0.7, Jsc=20.0, nid=1.5, Rs=1e-6, Rsh=1e6)
        V = np.linspace(-0.1, 0.75, 100)
        iv = IVData.from_fp(V, fp, perfparam=False)
        assert len(iv.x) == 100

    def test_perfparam_from_fp(self):
        fp = FiveParam(Voc=0.7, Jsc=20.0, nid=1.5, Rs=1e-6, Rsh=1e6)
        V = np.linspace(-0.1, 0.75, 200)
        iv = IVData.from_fp(V, fp, light_int=100.0, perfparam=True)
        assert iv.pd is not None
        assert iv.pd.Voc > 0


class TestDetVoc:
    def test_voc_within_1pct(self):
        true_voc = 0.70
        iv = _make_synthetic_jv(Voc=true_voc, Jsc=20.0, nid=1.5, Rsh=1e9)
        voc = iv.det_voc()
        assert abs(voc - true_voc) / true_voc < 0.01

    def test_voc_stored(self):
        iv = _make_synthetic_jv(Voc=0.65)
        voc = iv.det_voc()
        assert iv.Voc == voc

    def test_voc_alternative_method(self):
        iv = _make_synthetic_jv(Voc=0.70)
        voc = iv.det_voc(use_interpolate_extrapolate_method=False)
        assert abs(voc - 0.70) / 0.70 < 0.02


class TestDetJsc:
    def test_jsc_within_1pct(self):
        true_jsc = 20.0
        iv = _make_synthetic_jv(Voc=0.7, Jsc=true_jsc, nid=1.5, Rsh=1e9)
        jsc = iv.det_jsc()
        assert abs(jsc - true_jsc) / true_jsc < 0.01

    def test_jsc_stored(self):
        iv = _make_synthetic_jv(Jsc=18.0)
        jsc = iv.det_jsc()
        assert iv.Jsc == jsc

    def test_jsc_alternative_method(self):
        iv = _make_synthetic_jv(Jsc=20.0)
        jsc = iv.det_jsc(use_interpolate_extrapolate_method=False)
        assert abs(jsc - 20.0) / 20.0 < 0.02


class TestDetPerfparam:
    def test_perfparam_voc_jsc_ff(self):
        true_voc = 0.70
        true_jsc = 20.0
        iv = _make_synthetic_jv(Voc=true_voc, Jsc=true_jsc, nid=1.5, Rsh=1e9)
        iv.det_perfparam()
        pd_ = iv.pd
        assert abs(pd_.Voc - true_voc) / true_voc < 0.01
        assert abs(pd_.Jsc - true_jsc) / true_jsc < 0.01
        assert pd_.FF > 50.0
        assert pd_.FF < 100.0

    def test_pce_positive(self):
        iv = _make_synthetic_jv(Voc=0.7, Jsc=20.0)
        iv.det_perfparam()
        assert iv.pd.PCE > 0

    def test_perfparam_show(self):
        iv = _make_synthetic_jv(Voc=0.7, Jsc=20.0)
        iv.det_perfparam(show=True)
        plt.close('all')
        assert iv.pd is not None


class TestDetIni5Param:
    def test_det_ini_5param(self):
        iv = _make_synthetic_jv(Voc=0.7, Jsc=20.0, nid=1.5)
        iv.det_ini_5param()
        assert iv.Voc is not None
        assert iv.Jsc is not None


class TestIVDataSqLimit:
    def test_sq_limit_voc_positive(self):
        voc = IVData.sq_limit_voc(bg=1.6)
        assert voc > 0

    def test_sq_limit_jsc_positive(self):
        jsc = IVData.sq_limit_jsc(bg=1.6)
        assert jsc > 0

    def test_sq_limit_returns_fiveparam(self):
        fp = IVData.sq_limit(bg=1.6)
        assert isinstance(fp, FiveParam)
        assert fp.Voc > 0
        assert fp.Jsc > 0


class TestIVDataLoad:
    def test_load_csv(self):
        fp = os.path.join(FIXTURES, 'jv_curve.csv')
        iv = IVData.load(fp)
        assert len(iv.x) > 5
        assert iv.qx == 'Voltage'

    def test_det_voc_from_file(self):
        fp = os.path.join(FIXTURES, 'jv_curve.csv')
        iv = IVData.load(fp)
        voc = iv.det_voc()
        assert 0.65 < voc < 0.85


class TestLinearExtrapolate:
    def test_midpoint_interpolation(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 2.0, 4.0])
        result = IVData._linear_extrapolate(0.5, x, y)
        assert abs(result - 1.0) < 1e-10

    def test_extrapolation_below(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        result = IVData._linear_extrapolate(0.0, x, y)
        assert abs(result - 0.0) < 1e-10

    def test_extrapolation_above(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        result = IVData._linear_extrapolate(4.0, x, y)
        assert abs(result - 8.0) < 1e-10
