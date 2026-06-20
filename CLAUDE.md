# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scientific Python library for photovoltaic and energy materials analysis (EPFL / FTE group). Covers data import, processing, fitting, and visualization for JV curves, EIS, PL spectroscopy, TRPL, PLQY, redox-flow batteries, and related techniques.

Install locally for development:
```bash
pip install -e .
```

Install from parent directory (as described in README):
```bash
python -m pip install FTE-analysis-libraries-main
```

There are no tests and no linting configuration yet.

## Module overview

All modules live in `FTE_analysis_libraries/` and use PascalCase filenames. The dependency order (lower layers imported by higher ones):

| Module | Role |
|---|---|
| `General.py` | Physical constants, math helpers, plotting utilities, `df_interpolate`, `findind`, `save_ok` |
| `XYdata.py` | `xy_data` and `mxy_data` base classes — x/y arrays with units, quantity labels, plot styles, and methods for slicing, interpolation, plotting, and saving |
| `Spectrum.py` | Subclasses of `xy_data` for optical spectra; loads Andor/OceanOptics/Avantes files; spectral math (photon flux conversion, EQE, etc.) |
| `IV.py` | `IV_data` class and `fivep` dataclass; loads Biologic/Ossila/Zahner JV files; extracts Voc, Jsc, FF, PCE; SQ-limit calculations |
| `Electrochemistry.py` | CV/EIS loading (Biologic `.mpt`); `IV_data`-based analysis; uses `impedance` library for EIS circuit fitting |
| `PLQY.py` | Absolute PLQY calculation from integrating sphere spectra; uses `thot` for experiment management |
| `TRPL.py` | Time-resolved PL: multi-exponential fitting, IRF deconvolution, streak-camera data loading |
| `RFB.py` | Redox-flow battery: vanadium electrolyte calculations, cycling analysis, coulombic/energy efficiency; imports `Electrochemistry` |
| `Tkdialogs.py` | Tkinter file/directory dialog helpers |

`System_data/` holds reference spectra (AM1.5G, calibration lamp, luminosity function, SQ-limit Voc table) loaded via `pkg_resources.resource_filename`.

## Key patterns

**`xy_data` / `mxy_data`** — the central abstraction. `xy_data` wraps a single (x, y) pair; `mxy_data` wraps a list of `xy_data` objects (a "multi-spectrum"). Nearly every domain module subclasses or returns one of these.

**`save_ok(TFN)`** — interactive file-overwrite guard used before any `to_csv`. Call it before saving files.

**`findind(arr, value)`** — returns the index of the closest value in a monotone array. Relied on throughout for slicing.

**`df_interpolate(df, new_index_arr)`** — reindexes a DataFrame using quadratic interpolation. Used in spectral math and RFB cycling analysis.

**Physical constants** — defined in `General.py` (`q`, `k`, `T_RT`, `h`, `c`, `f1240`, etc.). Import from there; do not redefine.

## Planned refactoring

`FTE_analysis_libraries/CLAUDE.md` documents the target architecture (snake_case names, `pyproject.toml`, pytest, ruff, Sphinx). The current codebase is a pre-refactor snapshot. When working on the refactoring:
- Work in feature branches (`refactor/<topic>`, `feat/<topic>`, `fix/<topic>`)
- Follow Conventional Commits (`refactor: ...`, `feat: ...`, `fix: ...`)
- See `FTE_analysis_libraries/CLAUDE.md` for the full target code style and packaging plan
