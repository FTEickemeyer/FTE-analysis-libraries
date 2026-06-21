# Bug Tracker

Records all bugs found during the v1.0.0 modernisation audit, their status,
and remaining known issues to address in future work.

---

## Fixed in v1.0.0

### Critical bugs (runtime errors / wrong results)

| ID | File | Location | Description | Status |
|---|---|---|---|---|
| G-01/02 | General.py | `beep()` | `sys.platform == 'Win32'` is always `False`; `winsound` never imported → `NameError` on every call | Fixed |
| G-03 | General.py | `save_ok()` | Local variable `save_ok` shadowed the enclosing function name | Fixed |
| G-05 | General.py | `ignore_warnings()` | `return(func(...))` was **outside** the `with warnings.catch_warnings():` block — warning suppression was completely broken | Fixed |
| G-07 | General.py | `idx_range()` | Broken swap: `l = r; r = l` set both to `r`, discarding original `l` | Fixed |
| G-09 | General.py | module level | Constant `R = 0.99999999965e-3` ≈ 0.001; correct molar gas constant is 8.314 J/(mol·K) — wrong by ~4 orders of magnitude | Fixed |
| G-18 | General.py | `V_loss()` | `T` parameter accepted but body hard-coded `T_RT` — parameter silently ignored | Fixed |
| X-06 | XYdata.py | `plot_linfit()` | `return both` reached when `residue=True, return_data=True` but `both` was never assigned → `NameError` | Fixed |
| X-18 | XYdata.py | `xy_data.plot()`, `mxy_data.plot()` | When `hline` is a list and `hline_colors is None`, `color` was never assigned → `UnboundLocalError` | Fixed |
| X-28 | XYdata.py | `mxy_data.load_individual()` | `sp` only assigned inside `if cls.__name__ == 'mxy_data':` but referenced unconditionally → `UnboundLocalError` for subclasses | Fixed |
| X-39 | XYdata.py | `xy_data.__init__()` | `plotstyle = dict(...)` mutable default argument shared across all instances; `idfac_fit()` mutation corrupted the default for future instances | Fixed |
| S-55 | Spectrum.py | `BBT_fit()` | Bare `except:` caught `KeyboardInterrupt`/`SystemExit`; changed to `except RuntimeError:` | Fixed |
| S-59 | Spectrum.py | `abs_spectrum.calc_Vocrad()` | Referenced `self.Jradlim` set only by `calc_Jradlim()` — calling `calc_Vocrad` first raised `AttributeError` | Fixed |
| S-61 | Spectrum.py | `PEL_spectra.calc_calfn()` | Defined as `def calc_calfn(mspec, calspec):` inside a class — missing `self`; calling as instance method raised `TypeError` | Fixed |
| I-37 | IV.py | `load_Igor_IV()` | `cell_area`, `light_int`, `startV`, `deltaV`, `sweep_dir` only assigned inside keyword-conditional branches → `UnboundLocalError` if keywords absent | Fixed |
| TRPL-3 | TRPL.py | `plot_animation()` | `print#(int(...))` — `#` commented out the argument, printing a blank line | Fixed |
| TRPL-4 | TRPL.py | `plot_animation()` / `plot_animation_QFLS()` | Init checked `pset1.pulse_len == None` (Python None) but animate inner function checked `pset1.pulse_len == 'None'` (string) — one branch could never execute | Fixed |
| TRPL-7 | TRPL.py | `TRPL_data.load()` | Defined as regular method but first parameter was `directory` not `self` → `TypeError` on any instance call; fixed with `@staticmethod` | Fixed |
| TRPL-9 | TRPL.py | `TRPL_data.gen_m2ed()` | Accessed `p0[4]` but p0 is 4-element → `IndexError` | Fixed |
| EC-11 | Electrochemistry.py | `multiple_capacitance_plot` | `if __name__ == "__main__":` indented **inside** the function body → permanently unreachable code | Fixed |
| RFB-1 | RFB.py | `time_to_seconds` | `except: ValueError` — `ValueError` after the colon is an expression statement, not a caught type; no exception was actually caught | Fixed |
| RFB-2 | RFB.py | `split_dfcharging()` | Same `except: ValueError` anti-pattern | Fixed |
| RFB-5 | RFB.py | `upload_potential_difference_measurement()` | `FN_V` referenced in NI branch but never defined in this function → `NameError` | Fixed |
| PLQY-7 | PLQY.py | `exp_param.__init__()` | In `PL_peak=None` branch: `self.PL_peak` was never assigned → `AttributeError` on access | Fixed |
| PLQY-9 | PLQY.py | `PLQY_dataset.find_PL_peak()` | `PL_peak` local variable may never be assigned inside inner `if` but referenced on next line → `UnboundLocalError` | Fixed |
| TK-6 | Tkdialogs.py | `getFilenames()` | No cancellation guard: cancelled dialog returns `[]` and `filePaths[0]` raises `IndexError` | Fixed |
| TK-7 | Tkdialogs.py | `getFilename()` | No cancellation guard: cancelled dialog returns empty string silently | Fixed |

### Medium bugs (wrong behaviour, not always crashes)

| ID | File | Description | Status |
|---|---|---|---|
| G-04 | General.py | `linfit()` default `bis = array_y[-1]` — should be `array_x[-1]`; used y-array value as x-axis boundary | Fixed |
| G-06 | General.py | `how_long()` mutable default argument `arr = np.arange(1, 2, 1)` created once at definition time | Fixed |
| X-08 | XYdata.py | `== None` comparisons throughout instead of `is None` | Fixed |
| X-11 | XYdata.py | `polyfit()`: when `new_meshsize` is `None`, `None != 0` is `True` → `np.linspace(..., None)` raises `TypeError` | Fixed |
| X-17 | XYdata.py | `mxy_data.max_within()` initialised `maximum = 0` → returns 0 when all y-values are negative | Fixed |
| X-29 | XYdata.py | `from_df()`: `df.index.name.split(...)` raises `AttributeError` if index has no name | Fixed |
| I-36 | IV.py | `det_perfparam()`: `Pmpp = abs(Vmpp*Jmpp)` assigned twice with identical expression | Fixed |
| I-38 | IV.py | `ini_guess_Rsh()`: references `self.Jsc` before it is computed in `show_fit` path | Fixed |
| TRPL-8 | TRPL.py | `mult2_expfit`, `mult3_expfit`, `mult4_expfit` printed `"tau1"` for all time constants | Fixed |
| TRPL-14 | TRPL.py | `from_param()`: `time_delta = time_delta` self-assignment no-op | Fixed |
| EC-14/15 | Electrochemistry.py | `load_Biologic_CV` / `load_Biologic_CA`: `if light_int == None:` should be `is None` | Fixed |
| RFB-3 | RFB.py | `UVVIS_get_offset_and_scaling_factor_cuv_pos`: bare `except:` on `curve_fit` failure | Fixed |
| RFB-9 | RFB.py | `det_df_conc_from_charging()`: `df_conc[col] = df_conc[col].apply(lambda x: x)` — identity lambda, no-op | Fixed |
| RFB-12 | RFB.py | `fit_potentials()`: `c_V = c_V` self-assignment no-op | Fixed |
| PLQY-10 | PLQY.py | `inb_oob_adjust()`: `if adj_factor == None:` | Fixed |

### Python 3.14 compatibility

| File | Description | Status |
|---|---|---|
| IV.py | Invalid escape sequences `\;`, `\,`, `\%`, `\mu`, `\Omega`, `\infty`, `\ll`, `\Delta` in f-strings | Fixed |
| Electrochemistry.py | `\d` in regex string (`'(\d+)'` → `r'(\d+)'`) | Fixed |
| XYdata.py | `\cdot` in f-strings | Fixed |
| TRPL.py | Invalid escapes in docstrings | Fixed |
| PLQY.py | `\d` in four regex patterns | Fixed |
| RFB.py | LaTeX escapes in multi-line docstring string | Fixed |
| General.py | Escape sequence in path helper | Fixed |

---

## Remaining known issues

These were identified in the audit but not yet fixed. Grouped by priority.

### High priority

| ID | File | Location | Description |
|---|---|---|---|
| S-57 | Spectrum.py | `bg_from_ip()` | `if 'diff' in showplot:` raises `TypeError` when `showplot=None` — no default guard |
| S-58 | Spectrum.py | `diff_spectrum.nm_to_eV()` | `y#_eV = y[::-1] * ...` — `#` comments out `_eV`, turning the line into a dead expression; spectral conversion silently does nothing |
| X-12 | XYdata.py | `mxy_data.plot()` | `self = self.remain(...)` and `self = self_old` have no effect on the caller — Python cannot rebind `self`; related `self_old = self.copy()` is wasted work |
| RFB-14 | RFB.py | `__main__` block | Calls `conc_V_SO4(..., show_details=True)` but parameter is named `show` → `TypeError` |
| I-8 | IV.py | `table_param()` | No `self` parameter but lives inside the class — should be `@staticmethod` |

### Medium priority

| ID | File | Location | Description |
|---|---|---|---|
| X-40 | XYdata.py | `idfac_fit()` | `plotrange = [None, None]` is a mutable default list |
| X-07 | XYdata.py | `__init__()` | `type(quants) == list` should be `isinstance(quants, list)` |
| X-09/10 | XYdata.py | `all_values_greater_min` | Parameter `min` shadows Python builtin `min()` |
| X-12 | XYdata.py | `mxy_data.plot()` | `generate_image_stream` path saves to image stream *after* `plt.show()` — produces blank image |
| S-62 | Spectrum.py | `spectra.load_individual()` | Local variable `spectra = cls(sa)` shadows the class name `spectra` in classmethod scope |
| I-33–35 | IV.py | Throughout | Remaining `== None` comparisons |
| I-40 | IV.py | Three SQ methods | `== None` on `illumspec_PF_eV` — if a `diff_spectrum` is passed, comparison is element-wise |
| EC-7 | Electrochemistry.py | `EIS_convert_mpt_to_csv` | `temp.txt` written to disk with no `try/finally` guard — file leaked on any exception |
| EC-9 | Electrochemistry.py | `EIS_get_data_old()` | Loop-variable `f_idx_end` mutated inside loop; second iteration uses wrong slice (dead code, but still present) |
| RFB-11 | RFB.py | `UVVIS_fit_spectrum()` | First `bounds` assignment immediately overwritten on next line — dead statement |
| TK-2 | Tkdialogs.py | `__main__` block | `import Tkdialogs as tk` shadows module-level `import tkinter as tk` |

### Low priority / style

| Category | Description |
|---|---|
| Ruff | 2266 style issues reported by `ruff check .` (1499 auto-fixable with `ruff check --fix`) — mostly line length, naming, and import ordering in pre-existing code |
| Type hints | `# type: ignore` comments used as workarounds in several places where proper type narrowing would be cleaner |
| Dead code | `_old`-named methods were removed, but `plot_new()` / `load_old()` / `idfac_fit_old()` in XYdata.py still present |
| Docs build | `cd docs && make html` requires GNU Make; on Windows use `python -m sphinx docs/ docs/_build/html` instead |
| `G-23` | `plot_first_n_lines()` parameter named `dir` still shadows Python builtin `dir()` |
| `G-40` | `if __name__ == "__main__":` block appears mid-file in General.py; functions defined after it are invisible to that block |
| `G-28` | `interpolated_array` and `int_arr` in General.py do the same thing — one should wrap the other |
| `X-04` | hline/vline rendering block copy-pasted verbatim in `xy_data.plot` and `mxy_data.plot` |
| `I-14` | Three `SQ_limit_*` methods each repeat AM1.5 spectrum loading + scaling block |
