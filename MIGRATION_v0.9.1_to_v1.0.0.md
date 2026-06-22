# Migration notes: FTE-analysis-libraries v0.9.1 → v1.0.0

This file documents the breaking API changes introduced in v1.0.0 and the
migration work already done. Paste it into a new Claude Code session when
working on related issues.

---

## What changed

### 1. Package import path
```python
# old
from FTE_analysis_libraries import Spectrum as spc
# new
from fte_analysis_libraries import Spectrum as spc
```

### 2. Class renames

| Old | New |
|---|---|
| `xy_data` | `XYData` |
| `mxy_data` | `MXYData` |
| `xyz_data` | `XYZData` |
| `IV_data` | `IVData` |
| `mIV_data` | `MIVData` |
| `fivep` | `FiveParam` |
| `perf_dat` | `PerfData` |
| `TRPL_data` | `TRPLData` |
| `mTRPL_data` | `MTRPLData` |
| `TRPL_param` | `TRPLParam` |
| `spectrum` | `Spectrum` |
| `abs_spectrum` | `AbsSpectrum` |
| `diff_spectrum` | `DiffSpectrum` |
| `EQE_spectrum` | `EQESpectrum` |
| `PEL_spectrum` | `PELSpectrum` |
| `spectra` | `Spectra` |
| `abs_spectra` | `AbsSpectra` |
| `diff_spectra` | `DiffSpectra` |
| `PEL_spectra` | `PELSpectra` |
| `exp_param` | `ExpParam` |
| `PLQY_dataset` | `PLQYDataset` |

### 3. Module-level function renames (General.py)

| Old | New |
|---|---|
| `Diff_coeff` | `diff_coeff` |
| `Mobility` | `mobility` |
| `Vsq` | `v_sq` |
| `V_loss` | `v_loss` |
| `QFLS` | `qfls` |

### 3b. Module-level function renames (Electrochemistry.py)

| Old | New |
|---|---|
| `import_datum` | `import_biologic_mpt_data` |

### 3c. Method renames (MXYData / MTRPLData / Spectra)

| Old | New |
|---|---|
| `.add(item)` | `.append(item)` |

> **Note:** `.add()` cannot be word-boundary auto-migrated (conflicts with pandas `.add()`). It is handled via regex in `migrate_notebooks.py` and `update_notebooks_post_v1.py` — inspect `[review]` hits manually.

### 3d. Keyword argument renames (MXYData / XYData)

| Function | Old kwarg | New kwarg |
|---|---|---|
| `all_values_greater_min` | `min` | `min_val` |
| `XYData.load` / `MXYData.load` | `FN` | `filepath` |

### 4. Method renames

**IVData**
`det_Voc` → `det_voc`, `det_Jsc` → `det_jsc`, `det_J0` → `det_j0`,
`from_J0` → `from_j0`, `load_Igor_IV` → `load_igor_iv`,
`ini_guess_Rsh` → `ini_guess_rsh`, `ini_guess_nid_and_Rs` → `ini_guess_nid_and_rs`,
`IVsq` → `iv_sq`, `IVrad` → `iv_rad`, `IVtrans` → `iv_trans`,
`I_of_V` → `i_of_v`, `I_of_V_safe` → `i_of_v_safe`,
`SQ_limit_Voc` → `sq_limit_voc`, `SQ_limit_Jsc` → `sq_limit_jsc`,
`SQ_limit` → `sq_limit`

**PerfData**
`Jmpp_text` → `jmpp_text`, `Vmpp_text` → `vmpp_text`, `Pmpp_text` → `pmpp_text`,
`PCE_text` → `pce_text`, `FF_text` → `ff_text`, `Voc_text` → `voc_text`,
`Jsc_text` → `jsc_text`, `Rs_text` → `rs_text`, `Rsh_text` → `rsh_text`

**Spectrum / DiffSpectrum / EQESpectrum etc.**
`nm_to_eV` → `nm_to_ev`, `eV_to_nm` → `ev_to_nm`,
`load_Andor` → `load_andor`, `load_Cicci` → `load_cicci`,
`EQE100` → `eqe100`, `MMF` → `mmf`, `MMF_Eg` → `mmf_eg`, `MMF_test` → `mmf_test`,
`calc_Jsc` → `calc_jsc`, `normalize_to_Jsc` → `normalize_to_jsc`,
`U_energy_fit` → `u_energy_fit`, `new_UE` → `new_ue`,
`Tauc_plot` → `tauc_plot`, `calc_Vocrad` → `calc_vocrad`,
`calc_Jradlim` → `calc_jradlim`, `load_ASTMG173` → `load_astmg173`,
`load_OSRAM930` → `load_osram930`, `load_LED5000K` → `load_led5000k`,
`AM15_nm` → `am15_nm`, `AM15_eV` → `am15_ev`,
`AM15direct_nm` → `am15direct_nm`, `AM15direct_eV` → `am15direct_ev`,
`Etr_nm` → `etr_nm`, `Etr_eV` → `etr_ev`,
`PhiBB` → `phi_bb`, `BBspectrum` → `bb_spectrum`, `BBT_fit` → `bbt_fit`,
`choose_for_PLQY` → `choose_for_plqy`, `calc_PLQY_param` → `calc_plqy_param`,
`Udata_plot` → `udata_plot`

---

## Migration already applied

The script `migrate_notebooks.py` (in the project root) was run on
`C:\Users\dreickem\switchdrive\Work\Python` in June 2026 and updated
**1,757 notebooks and .py files** automatically. Every modified file has a
`.bak_v091` backup alongside it.

### Directories intentionally excluded (not migrated)
- `My_packages\260429 FTE-analysis-libraries-main\` — old library snapshot
- `My modules_\` — older copies of library modules
- `My_packages\FTE-impedance-main\` and `..._old\`, `impedance.py-main_original\` — third-party
- `Laboratory\pymeasure\` — third-party instrument package
- `old\UW\Ryan\PVtools-master\` — Ryan's own code
- `PV tools\SQ limit\llight_spectra\Sandy\` — Sandy's code

---

## Known false-positive patterns

The migration script uses word-boundary regex, so `spectrum` and `spectra`
(common English/pvlib/pymeasure identifiers) occasionally get renamed
incorrectly. Symptoms: `ImportError` or `AttributeError` on a name that was
never part of the FTE library.

**Confirmed false positive:**
- `PV tools\Huawei_Stephanie_simulator\fit_JV_curves\get_data.py`
  — imports `from pvlib import spectrum` (lowercase); was incorrectly renamed
  to `Spectrum`. Restored from backup.

**Other files to watch** (only `[review]` hits, not verified):
- `PV tools\Huawei_Stephanie_simulator\` (batch_run, validation_parameters, old\)
- `VIS map\visiblespectrum.py`, `visiblespectrum_demo2.py`

### How to restore any file from backup
```powershell
Copy-Item "path\to\file.py.bak_v091" "path\to\file.py" -Force
# or for notebooks:
Copy-Item "path\to\notebook.ipynb.bak_v091" "path\to\notebook.ipynb" -Force
```

---

## Other bugs fixed in v1.0.0 (already in the library)

- **NumPy < 2.0 compatibility**: `np.trapezoid` (added in NumPy 2.0) is
  patched onto `np` in `General.py` if absent, so the library works on
  NumPy 1.x. Fixed in commit `c4bc09e`.

---

## Re-running the migration script

To migrate any new notebooks/scripts written against v0.9.1:

```powershell
cd "C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-analysis-libraries-main"
python migrate_notebooks.py          # dry run
python migrate_notebooks.py --apply  # apply (creates .bak_v091 backups)
```
