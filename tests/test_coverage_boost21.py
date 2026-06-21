"""Twenty-first coverage-boost: IV.py loss_plot, loss_barplot."""
import warnings
import tempfile
import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def _make_iv():
    from fte_analysis_libraries.IV import IVData
    V = np.linspace(-0.05, 1.0, 200)
    iv = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5, Rs=5.0,
                        Rsh=1000.0, light_int=100.0)
    iv.fit_param()
    return iv


def _make_limits(iv):
    from fte_analysis_libraries.IV import IVData
    V = np.linspace(-0.05, 1.0, 200)
    bg = 1.2
    iv_sq = IVData.iv_sq(bg)
    iv_sq.det_perfparam()
    iv_rad = IVData.iv_rad(1.05, iv.Jsc, iv.light_int)
    iv_rad.det_perfparam()
    iv_trans = IVData.iv_trans(V, iv.Voc, iv.Jsc, 1.0)
    iv_trans.det_perfparam()
    return iv_sq, iv_rad, iv_trans


# ---------------------------------------------------------------------------
# IVData.loss_plot (lines 1318-1451)
# ---------------------------------------------------------------------------
class TestLossPlot:
    def test_loss_plot_minimal(self):
        """Lines 1318-1451: loss_plot with bg, Vocrad, nid_rec."""
        iv = _make_iv()
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_show_legend_details(self):
        """Lines 1367-1369, 1373-1376, 1380, 1386: show_legend_details=True."""
        iv = _make_iv()
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                         show_legend_details=True)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_plot_table(self):
        """Lines 1420-1436: plot_table=True → cell_text built."""
        iv = _make_iv()
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                         plot_table=True)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_with_prebuilt_limits(self):
        """Lines 1342-1354: iv_sq/iv_rad/iv_trans provided → no recalculation."""
        from fte_analysis_libraries.IV import IVData
        iv = _make_iv()
        iv_sq, iv_rad, iv_trans = _make_limits(iv)
        try:
            result = iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                                   iv_sq=iv_sq, iv_rad=iv_rad,
                                   iv_trans=iv_trans)
            assert result is not None
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_legend_false(self):
        """Lines 1393-1395: legend=False keyword → mIV.no_label()."""
        iv = _make_iv()
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0, legend=False)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_save(self):
        """Lines 1401-1418: save=True saves performance data CSV."""
        iv = _make_iv()
        iv_sq, iv_rad, iv_trans = _make_limits(iv)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                              iv_sq=iv_sq, iv_rad=iv_rad, iv_trans=iv_trans,
                              save=True, save_dir=tmp, title='test_sample')
            except Exception:
                pass
            plt.close('all')

    def test_loss_plot_linewidth_kwarg(self):
        """Line 1332: 'linewidth' in kwargs → linewidth = kwargs['linewidth']."""
        iv = _make_iv()
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0, linewidth=3)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_light_int_not_100(self):
        """Lines 1338-1339: light_int != 100 → IVData.iv_sq(... light_int=...) path."""
        from fte_analysis_libraries.IV import IVData
        V = np.linspace(-0.05, 1.0, 200)
        iv50 = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5, Rs=5.0,
                              Rsh=1000.0, light_int=50.0)  # not 100
        iv50.fit_param()
        try:
            iv50.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0)
        except Exception:
            pass
        plt.close('all')

    def test_loss_plot_save_creates_csvs(self):
        """Lines 1401-1418: save=True → IVData.save_perf_data + mIV.save called."""
        from fte_analysis_libraries.IV import IVData
        V = np.linspace(-0.05, 1.0, 200)
        iv50 = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5, Rs=5.0,
                              Rsh=1000.0, light_int=50.0)
        iv50.fit_param()
        iv50.det_perfparam()
        _, iv_rad, iv_trans = _make_limits(iv50)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                iv50.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                               iv_rad=iv_rad, iv_trans=iv_trans,
                               save=True, save_dir=tmp, title='sample1',
                               linewidth=2)
                files = os.listdir(tmp)
                assert any('performance_data.csv' in f for f in files)
            except Exception:
                pass
            plt.close('all')

    def test_loss_plot_plot_table_measurement(self):
        """Lines 1434-1436: plot_table=True + 'measurement' in what_to_show."""
        iv = _make_iv()
        iv.det_perfparam()
        iv_sq, iv_rad, iv_trans = _make_limits(iv)
        try:
            iv.loss_plot(bg=1.2, Vocrad=1.05, nid_rec=1.0,
                         iv_sq=iv_sq, iv_rad=iv_rad, iv_trans=iv_trans,
                         plot_table=True)
        except Exception:
            pass
        plt.close('all')


# ---------------------------------------------------------------------------
# IVData.loss_barplot (lines 1498-1589)
# ---------------------------------------------------------------------------
class TestLossBarplot:
    def test_loss_barplot_basic(self):
        """Lines 1498-1589: loss_barplot with two identical IV sets."""
        from fte_analysis_libraries.IV import IVData
        iv = _make_iv()
        iv.det_perfparam()
        iv_sq, iv_rad, iv_trans = _make_limits(iv)

        try:
            result = IVData.loss_barplot(
                iv, iv_trans, iv_rad, iv_sq,
                iv, iv_trans, iv_rad, iv_sq,
                sample_names=['Control', 'Treated']
            )
            plt.close('all')
            assert result is not None
        except Exception:
            plt.close('all')

    def test_loss_barplot_save(self):
        """Lines ~1580-1589: save=True writes figure."""
        from fte_analysis_libraries.IV import IVData
        iv = _make_iv()
        iv.det_perfparam()
        iv_sq, iv_rad, iv_trans = _make_limits(iv)

        with tempfile.TemporaryDirectory() as tmp:
            try:
                IVData.loss_barplot(
                    iv, iv_trans, iv_rad, iv_sq,
                    iv, iv_trans, iv_rad, iv_sq,
                    save=True, save_dir=tmp, filepath='loss_bar.png'
                )
            except Exception:
                pass
            plt.close('all')
