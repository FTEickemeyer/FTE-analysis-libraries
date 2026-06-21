"""Twentieth coverage-boost: TRPL plot_animation and plot_animation_QFLS."""
import warnings
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def _make_pset(pulse_len=None):
    from fte_analysis_libraries.TRPL import TRPLParam
    # Pass N_points=5 to constructor so that p.x is also length-5
    p = TRPLParam(pulse_len=pulse_len, N_points=5)
    p.finaltime = 0.5e-9
    p.dt = 1e-11
    p.n0 = np.ones(5) * 1e14
    return p


def _fake_func_animation(fig, func, frames=None, interval=None,
                          init_func=None, blit=None, repeat=None, **kw):
    """Fake FuncAnimation that actually calls init and animate once."""
    from unittest.mock import MagicMock
    if init_func is not None:
        init_func()     # covers init() body lines
    if func is not None:
        func(0)         # covers animate(i) body lines
    return MagicMock()


# ---------------------------------------------------------------------------
# plot_animation (lines 235-329) including inner init/animate bodies
# ---------------------------------------------------------------------------
class TestPlotAnimation:
    def _run(self, p1, p2, **kwargs):
        import matplotlib.animation as anim_module
        from unittest.mock import patch
        from fte_analysis_libraries.TRPL import plot_animation

        with patch.object(anim_module, 'FuncAnimation', _fake_func_animation):
            result = plot_animation(p1, p2, **kwargs)
        plt.close('all')
        return result

    def test_plot_animation_pulse_len_none(self):
        """Lines 235-329 (incl. init/animate bodies): pulse_len=None path."""
        p1 = _make_pset(pulse_len=None)
        p2 = _make_pset(pulse_len=None)
        result = self._run(p1, p2)
        assert result is not None

    def test_plot_animation_pulse_len_set(self):
        """Lines 247, 298: pulse_len != None → u1=zeros and pulse() in animate."""
        p1 = _make_pset(pulse_len=60e-12)
        p2 = _make_pset(pulse_len=60e-12)
        result = self._run(p1, p2)
        assert result is not None

    def test_plot_animation_normalize_to_end(self):
        """Lines 313-315: normalize_to_end=True in animate() inner function."""
        p1 = _make_pset(pulse_len=None)
        p2 = _make_pset(pulse_len=None)
        result = self._run(p1, p2, normalize_to_end=True)
        assert result is not None


# ---------------------------------------------------------------------------
# plot_animation_QFLS (lines 360-454) including inner init/animate bodies
# ---------------------------------------------------------------------------
class TestPlotAnimationQLFS:
    def _run(self, p1, p2, **kwargs):
        import matplotlib.animation as anim_module
        from unittest.mock import patch
        from fte_analysis_libraries.TRPL import plot_animation_QFLS

        with patch.object(anim_module, 'FuncAnimation', _fake_func_animation):
            result = plot_animation_QFLS(p1, p2, **kwargs)
        plt.close('all')
        return result

    def test_plot_animation_qfls_pulse_len_none(self):
        """Lines 360-454 (incl. init/animate bodies): pulse_len=None path."""
        p1 = _make_pset(pulse_len=None)
        p2 = _make_pset(pulse_len=None)
        self._run(p1, p2)

    def test_plot_animation_qfls_pulse_len_set(self):
        """Lines 371-372, 419-450: pulse_len != None path in init and animate."""
        p1 = _make_pset(pulse_len=60e-12)
        p2 = _make_pset(pulse_len=60e-12)
        self._run(p1, p2)
