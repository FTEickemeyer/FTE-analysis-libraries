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

## v1.0.0 migration — ongoing fixes

The package was renamed from `FTE_analysis_libraries` (v0.9.1) to `fte_analysis_libraries` (v1.0.0) with many class, function, and parameter renames. A bulk migration was applied to ~1,800 notebooks and scripts in June 2026. New errors still surface as notebooks are run for the first time.

### Triage workflow for new errors

1. **AttributeError / TypeError on a known class or function** — likely a rename missed by migration. Check `MIGRATION_v0.9.1_to_v1.0.0.md` for the new name, then add the rename to `update_notebooks_post_v3.py` (or a new `post_vN.py`) and to `migrate_notebooks.py`.

2. **NameError / broken logic inside a library function** — likely a function body gutted by the ruff auto-fix pass (see warning below). Compare the current code against git history (`git show 71d9a91:FTE_analysis_libraries/<Module>.py`) to recover the original logic.

3. **New rename not yet in any script** — add it to both the current `post_vN.py` and `migrate_notebooks.py` so full reruns stay consistent.

### Migration scripts

| Script | Covers | Backup suffix |
|---|---|---|
| `migrate_notebooks.py` | All v0.9.1→v1.0.0 renames (use for full reruns) | `.bak_v091` |
| `update_notebooks_post_v1.py` | `import_datum→import_biologic_mpt_data`, `.add()→.append()` | `.bak_post_v1` |
| `update_notebooks_post_v2.py` | `all_values_greater_min(min=)→(min_val=)` | `.bak_post_v2` |
| `update_notebooks_post_v3.py` | `, FN=→, filepath=` in `.load()` calls | `.bak_post_v3` |

Run any script without arguments for a dry run; add `--apply` to write changes.

### Warning: ruff-gutted function bodies

The ruff auto-fix pass (commit `f9be141`) silently emptied at least one function body: `PLQYDataset.inb_oob_adjust` in `PLQY.py` (restored in commit `5ed00ef`). If a library function raises `NameError` on a variable that should have been computed internally, check whether its body was similarly truncated. Use `git show 71d9a91:FTE_analysis_libraries/<Module>.py` to see the original.

## Planned refactoring

`FTE_analysis_libraries/CLAUDE.md` documents the target architecture (snake_case names, `pyproject.toml`, pytest, ruff, Sphinx). The current codebase is a pre-refactor snapshot. When working on the refactoring:
- Work in feature branches (`refactor/<topic>`, `feat/<topic>`, `fix/<topic>`)
- Follow Conventional Commits (`refactor: ...`, `feat: ...`, `fix: ...`)
- See `FTE_analysis_libraries/CLAUDE.md` for the full target code style and packaging plan
