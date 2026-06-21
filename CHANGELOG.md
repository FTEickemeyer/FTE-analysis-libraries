# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-06-21

First versioned release. The codebase was a working but informal research
script collection. This release modernises it into a pip-installable,
tested, and documented library without breaking existing functionality.

### Bug Fixes

- Fixed 25 critical runtime bugs across all modules, including:
  - `General.py`: wrong gas constant `R` (~0.001 instead of 8.314), broken
    `ignore_warnings()` indentation, `idx_range()` swap logic, `beep()` never
    imported on Windows (`sys.platform == 'Win32'` always `False`)
  - `XYdata.py`: `UnboundLocalError` in `hline` rendering when `hline_colors`
    is `None`; mutable default `plotstyle` dict shared across instances;
    `max_within()` initialised to `0` instead of `-inf`
  - `Spectrum.py`: bare `except:` in `BBT_fit` catching `KeyboardInterrupt`;
    `calc_calfn` missing `self` parameter; `bg_from_ip` TypeError on
    `showplot=None`
  - `IV.py`: `UnboundLocalError` in `load_Igor_IV` when header keywords absent
  - `TRPL.py`: `TRPL_data.load` missing `@staticmethod`; `gen_m2ed` accessing
    `p0[4]` on a 4-element tuple
  - `RFB.py`: `except: ValueError` anti-pattern (exception not caught);
    `FN_V` `NameError` in NI measurement branch
  - `PLQY.py`: `self.PL_peak` never assigned in `None` branch;
    `UnboundLocalError` in `find_PL_peak`
  - `Tkdialogs.py`: no cancellation guard — cancelled dialog raises `IndexError`
- Fixed medium-priority bugs: `== None` comparisons replaced with `is None`
  throughout; mutable default arguments; `polyfit` `None != 0` path
- Fixed all Python 3.14 `SyntaxWarning`s from invalid escape sequences
  (`\;`, `\,`, `\%`, `\mu`, `\infty`, `\cdot`, `\d`) in matplotlib label
  strings and regex patterns across `IV.py`, `Electrochemistry.py`,
  `XYdata.py`, `TRPL.py`, `PLQY.py`, `RFB.py`, `General.py`

### Refactoring

- Removed 18+ `_old`-named dead methods across all modules
- Removed unused imports (`sys`, `reload`, `embed`, duplicate `matplotlib`)
- Standardised naming to PEP 8: snake_case functions/variables, CamelCase
  classes (`fivep` → `FiveParam`, `perf_dat` → `PerfData`, etc.)
- Extracted duplicate helpers:
  - `XYdata.py`: shared `_hline_vline`, `_bottom_top_for_plot`, arithmetic
    `_align` helper
  - `TRPL.py`: shared `_expfit_core` for the four `mult*_expfit` methods
  - `Electrochemistry.py`: module-level `_read_mpt_header_lines` replacing
    three identical nested `header_lines()` definitions
- Replaced `dir` and `FN` parameter names (shadow builtins) throughout

### New Features

- `pyproject.toml` — pip-installable packaging via Hatchling (PEP 517/518)
  with `[project]` metadata, `[project.optional-dependencies]` dev group,
  `[project.urls]`, authors, keywords, and classifiers
- Type hints added to all public functions and class signatures across all
  modules (`Any`, `float | None`, `np.ndarray`, etc.)
- `[tool.ruff]` and `[tool.mypy]` configuration added to `pyproject.toml`

### Documentation

- NumPy-style docstrings added to every public function and class across all
  modules (Parameters, Returns, Examples sections)
- Sphinx documentation set up in `docs/` with `autodoc`, `numpydoc`, and
  `sphinx_rtd_theme`; builds with 0 warnings (`docs/_build/html/`)
- API reference pages for all 9 modules: General, XYdata, Spectrum, IV,
  Electrochemistry, TRPL, PLQY, RFB, Tkdialogs
- `README.md` rewritten with library description, `pip install` instructions
  (from GitHub with version tag), quick usage example, and module table
- `CLAUDE.md` added for Claude Code sessions

### Testing

- pytest test suite introduced: 793 tests, 67% line coverage
- Coverage highlights: `General.py` 96%, `XYdata.py` 74%, `Spectrum.py` 100%,
  `IV.py` 94%, `TRPL.py` 100%
- Hardware/instrument modules (`RFB`, `Electrochemistry`, `PLQY`,
  `Tkdialogs`) remain at lower coverage — require physical devices or
  proprietary file formats unavailable in CI

---

## [0.0.2] — pre-release snapshot

Internal snapshot before the v1.0.0 modernisation effort began.
Working research-script collection, not pip-installable.
