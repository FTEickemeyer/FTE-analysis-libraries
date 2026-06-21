"""Tests for RFB.py — vanadium redox-flow battery calculations."""
import numpy as np
import pytest

from fte_analysis_libraries.RFB import (
    calc_conc_functions, calc_df_conc, conc_V_SO4,
)


class TestCalcConcFunctions:
    def test_returns_two_callables(self):
        Vconc_x_fn, Xconc_Vconc_fn = calc_conc_functions(c_V=1.6, c_SO4=4.0)
        assert callable(Vconc_x_fn)
        assert callable(Xconc_Vconc_fn)

    def test_vanadium_concentrations_at_x2(self):
        Vconc_x_fn, _ = calc_conc_functions(c_V=1.6, c_SO4=4.0)
        conc = Vconc_x_fn(2.0)
        # At fully-reduced state (x=2): all V2+, none of others
        assert abs(conc['c_2'] - 1.6) < 1e-10
        assert conc['c_3'] == 0
        assert conc['c_4'] == 0
        assert conc['c_5'] == 0

    def test_vanadium_concentrations_at_x5(self):
        Vconc_x_fn, _ = calc_conc_functions(c_V=1.6, c_SO4=4.0)
        conc = Vconc_x_fn(5.0)
        # At fully-oxidized state (x=5): all V5+
        assert abs(conc['c_5'] - 1.6) < 1e-10
        assert conc['c_2'] == 0
        assert conc['c_3'] == 0
        assert conc['c_4'] == 0

    def test_vanadium_concentrations_at_x3(self):
        Vconc_x_fn, _ = calc_conc_functions(c_V=2.0, c_SO4=4.0)
        conc = Vconc_x_fn(3.0)
        # At x=3: all V3+
        assert abs(conc['c_3'] - 2.0) < 1e-10
        assert conc['c_2'] == 0

    def test_vanadium_concentrations_at_x4(self):
        Vconc_x_fn, _ = calc_conc_functions(c_V=1.5, c_SO4=3.5)
        conc = Vconc_x_fn(4.0)
        # At x=4: all VO2+
        assert abs(conc['c_4'] - 1.5) < 1e-10

    def test_sum_is_total_vanadium(self):
        c_V = 1.6
        Vconc_x_fn, _ = calc_conc_functions(c_V=c_V, c_SO4=4.0)
        for x in [2.5, 3.0, 3.5, 4.0, 4.5]:
            conc = Vconc_x_fn(x)
            total = conc['c_2'] + conc['c_3'] + conc['c_4'] + conc['c_5']
            assert abs(total - c_V) < 1e-10

    def test_proton_conc_positive(self):
        c_V = 1.6
        _, Xconc_Vconc_fn = calc_conc_functions(c_V=c_V, c_SO4=4.0)
        Vconc_x_fn, _ = calc_conc_functions(c_V=c_V, c_SO4=4.0)
        conc = Vconc_x_fn(3.5)
        sol = Xconc_Vconc_fn(conc)
        # sol[3] = [H+], must be positive
        assert sol[3] > 0


class TestCalcDfConc:
    def test_returns_dataframe(self):
        import pandas as pd
        df = calc_df_conc(c_V=1.6, c_SO4=4.0)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = calc_df_conc(c_V=1.6, c_SO4=4.0)
        # calc_df_conc returns sulfate/proton species columns
        assert len(df.columns) > 0

    def test_non_empty(self):
        df = calc_df_conc(c_V=1.6, c_SO4=4.0)
        assert len(df) > 0


class TestConcVSO4:
    def test_returns_tuple(self):
        result = conc_V_SO4()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_positive(self):
        c_V, c_SO4 = conc_V_SO4()
        assert c_V > 0
        assert c_SO4 > 0
