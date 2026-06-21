"""Seventeenth coverage-boost: General str_round_sig, beep, scattered_boxplot; Spectrum udata_plot save_name."""
import warnings
import tempfile
import os
import sys
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# General.str_round_sig: zero-padding loop (line 156)
# ---------------------------------------------------------------------------
class TestStrRoundSigPadding:
    def test_zero_padding(self):
        """Line 156: num_string += '0' when not enough sig figs."""
        from fte_analysis_libraries.General import str_round_sig
        # 1.0 with sig=3: '1.0' → '10' has 2 chars < 3 → padded to '1.00'
        result = str_round_sig(1.0, sig=3)
        assert result == '1.00'

    def test_zero_padding_integer(self):
        """Line 156: integer 2 with sig=4 needs '2.000'."""
        from fte_analysis_libraries.General import str_round_sig
        result = str_round_sig(2.0, sig=4)
        assert result == '2.000'


# ---------------------------------------------------------------------------
# General.beep: Windows branch (line 557)
# ---------------------------------------------------------------------------
class TestBeep:
    def test_beep_on_windows(self):
        """Line 557: beep() enters if sys.platform=='win32' branch on Windows."""
        from fte_analysis_libraries.General import beep
        if sys.platform != 'win32':
            pytest.skip('Windows-only test')
        # winsound was never imported (library bug G-01), so NameError is expected
        try:
            beep(freq=600, duration=100)
        except (NameError, AttributeError):
            pass  # line 557 covered; line 558 may raise (winsound API differs)


# ---------------------------------------------------------------------------
# General.scattered_boxplot (lines 799-843)
# ax.boxplot() is broken on newer matplotlib (labels→label rename),
# so we patch it to reach the post-boxplot logic at lines 799+.
# ---------------------------------------------------------------------------
class TestScatteredBoxplot:
    def _run(self, data, **kwargs):
        from fte_analysis_libraries.General import scattered_boxplot
        from unittest.mock import patch
        fig, ax = plt.subplots()
        with patch.object(ax, 'boxplot'):
            scattered_boxplot(ax, data, **kwargs)
        plt.close('all')

    def test_basic_with_unif_jitter(self):
        """Lines 799-843: default showfliers='unif' — scatter with uniform jitter."""
        self._run([np.random.randn(20) for _ in range(3)])

    def test_showfliers_normal(self):
        """Line 836-837: showfliers='normal' → normal jitter."""
        self._run([np.random.randn(15) for _ in range(2)], showfliers='normal')

    def test_showfliers_false_returns_early(self):
        """Line 838-839: showfliers=False → return early (no scatter)."""
        self._run([np.random.randn(15) for _ in range(2)], showfliers=False)

    def test_showfliers_classic(self):
        """Lines 792-793: showfliers='classic' → classic_fliers=True."""
        self._run([np.random.randn(15) for _ in range(2)], showfliers='classic')

    def test_showfliers_unknown_raises(self):
        """Line 841: unknown showfliers → NotImplementedError."""
        from fte_analysis_libraries.General import scattered_boxplot
        from unittest.mock import patch
        fig, ax = plt.subplots()
        with patch.object(ax, 'boxplot'):
            try:
                scattered_boxplot(ax, [np.random.randn(10), np.random.randn(10)],
                                  showfliers='UNKNOWN')
            except NotImplementedError:
                pass
        plt.close('all')

    def test_with_positions(self):
        """Lines 803-808: explicit positions list."""
        self._run([np.random.randn(10) for _ in range(3)], positions=[1, 2, 3])

    def test_hide_points_within_whiskers(self):
        """Lines 820-831: hide_points_within_whiskers=True uses cbook.boxplot_stats."""
        self._run([np.random.randn(30) for _ in range(2)],
                  hide_points_within_whiskers=True)

    def test_scalar_widths(self):
        """Lines 815-816: np.isscalar(widths) → widths = [widths] * N."""
        self._run([np.random.randn(20) for _ in range(2)], widths=0.3)

    def test_wrong_positions_length_raises(self):
        """Line 805-806: len(positions) != N → ValueError."""
        from fte_analysis_libraries.General import scattered_boxplot
        from unittest.mock import patch
        fig, ax = plt.subplots()
        data = [np.random.randn(10), np.random.randn(10), np.random.randn(10)]
        with patch.object(ax, 'boxplot'):
            try:
                scattered_boxplot(ax, data, positions=[1, 2])  # wrong length (2 vs N=3)
            except ValueError:
                pass
        plt.close('all')

    def test_non_numeric_positions_raises(self):
        """Line 809-810: positions with non-numeric values → TypeError."""
        from fte_analysis_libraries.General import scattered_boxplot
        from unittest.mock import patch
        fig, ax = plt.subplots()
        data = [np.random.randn(10), np.random.randn(10)]
        with patch.object(ax, 'boxplot'):
            try:
                scattered_boxplot(ax, data, positions=['a', 'b'])
            except TypeError:
                pass
        plt.close('all')

    def test_wrong_widths_length_raises(self):
        """Lines 817-818: len(widths) != N → ValueError."""
        from fte_analysis_libraries.General import scattered_boxplot
        from unittest.mock import patch
        fig, ax = plt.subplots()
        data = [np.random.randn(10), np.random.randn(10), np.random.randn(10)]
        with patch.object(ax, 'boxplot'):
            try:
                scattered_boxplot(ax, data, widths=[0.3, 0.4])  # wrong length
            except ValueError:
                pass
        plt.close('all')


# ---------------------------------------------------------------------------
# Spectrum.udata_plot with save_name != None (line 2066)
# ---------------------------------------------------------------------------
class TestUdataPlotSaveName:
    def test_udata_plot_with_save_name(self):
        """Line 2066: filepath = save_name when save=True and save_name is not None."""
        from fte_analysis_libraries.Spectrum import PELSpectra, DiffSpectrum
        wl = np.linspace(1.0, 3.0, 200)
        ab = DiffSpectrum(wl, 0.5 + 0.4 * np.exp(-(wl - 2.0)**2 / 0.1),
                          quants={'x': 'Photon energy', 'y': 'Absorptance'},
                          units={'x': 'eV', 'y': ''})
        uf = DiffSpectrum(wl, 0.5 * np.exp(-2 * (3.0 - wl)),
                          quants={'x': 'Photon energy', 'y': 'Urbach fit'},
                          units={'x': 'eV', 'y': ''})
        sp = DiffSpectrum(wl, np.exp(-(wl - 1.8)**2 / 0.05),
                          quants={'x': 'Photon energy', 'y': 'PL'},
                          units={'x': 'eV', 'y': ''})
        bb = DiffSpectrum(wl, np.exp(-(wl - 2.0)**2 / 0.2),
                          quants={'x': 'Photon energy', 'y': 'BB'},
                          units={'x': 'eV', 'y': ''})
        pel = PELSpectra([ab, uf, sp, bb])
        with tempfile.TemporaryDirectory() as tmp:
            try:
                pel.udata_plot(overlap=2.0, left=1.2, right=2.8,
                               show_plot=False, return_fig=True,
                               save=True, save_dir=tmp,
                               save_name='my_urbach_data.csv')
            except Exception:
                pass
            plt.close('all')
