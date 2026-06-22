"""Migrate Jupyter notebooks from FTE-analysis-libraries v0.9.1 to v1.0.0 API.

Usage:
    python migrate_notebooks.py           # dry run (shows what would change)
    python migrate_notebooks.py --apply   # apply changes (creates .bak_v091 backups)
"""

import json
import re
import sys
from pathlib import Path

NOTEBOOKS_ROOT = Path(r"C:\Users\dreickem\switchdrive\Work\Python")

# (old, new) — ordered most-specific first to avoid double-replacement.
REPLACEMENTS = [
    # Package import path
    ("FTE_analysis_libraries", "fte_analysis_libraries"),

    # Class renames — compound before simple
    ("mxy_data",        "MXYData"),
    ("xyz_data",        "XYZData"),
    ("xy_data",         "XYData"),
    ("mIV_data",        "MIVData"),
    ("IV_data",         "IVData"),
    ("mTRPL_data",      "MTRPLData"),
    ("TRPL_data",       "TRPLData"),
    ("TRPL_param",      "TRPLParam"),
    ("fivep",           "FiveParam"),
    ("perf_dat",        "PerfData"),
    ("exp_param",       "ExpParam"),
    ("PLQY_dataset",    "PLQYDataset"),
    ("abs_spectra",     "AbsSpectra"),
    ("diff_spectra",    "DiffSpectra"),
    ("PEL_spectra",     "PELSpectra"),
    ("abs_spectrum",    "AbsSpectrum"),
    ("diff_spectrum",   "DiffSpectrum"),
    ("EQE_spectrum",    "EQESpectrum"),
    ("PEL_spectrum",    "PELSpectrum"),
    # "spectra" / "spectrum" are common English words; replaced with word
    # boundaries but flagged for review in the output.
    ("spectra",         "Spectra"),
    ("spectrum",        "Spectrum"),

    # Module-level functions (General.py)
    ("Diff_coeff",      "diff_coeff"),
    ("Mobility",        "mobility"),
    ("Vsq",             "v_sq"),
    ("V_loss",          "v_loss"),
    ("QFLS",            "qfls"),

    # IVData methods — longer names first
    ("ini_guess_nid_and_Rs",    "ini_guess_nid_and_rs"),
    ("ini_guess_Rsh",           "ini_guess_rsh"),
    ("load_Igor_IV",            "load_igor_iv"),
    ("I_of_V_safe",             "i_of_v_safe"),
    ("I_of_V",                  "i_of_v"),
    ("SQ_limit_Voc",            "sq_limit_voc"),
    ("SQ_limit_Jsc",            "sq_limit_jsc"),
    ("SQ_limit",                "sq_limit"),
    ("IVsq",                    "iv_sq"),
    ("IVrad",                   "iv_rad"),
    ("IVtrans",                 "iv_trans"),
    ("det_Voc",                 "det_voc"),
    ("det_Jsc",                 "det_jsc"),
    ("det_J0",                  "det_j0"),
    ("from_J0",                 "from_j0"),

    # PerfData methods
    ("Jmpp_text",   "jmpp_text"),
    ("Vmpp_text",   "vmpp_text"),
    ("Pmpp_text",   "pmpp_text"),
    ("PCE_text",    "pce_text"),
    ("FF_text",     "ff_text"),
    ("Voc_text",    "voc_text"),
    ("Jsc_text",    "jsc_text"),
    ("Rs_text",     "rs_text"),
    ("Rsh_text",    "rsh_text"),

    # Electrochemistry functions
    ("import_datum",         "import_biologic_mpt_data"),

    # Spectrum methods — longer / more specific first
    ("normalize_to_Jsc",    "normalize_to_jsc"),
    ("load_ASTMG173",       "load_astmg173"),
    ("load_OSRAM930",       "load_osram930"),
    ("load_LED5000K",       "load_led5000k"),
    ("AM15direct_nm",       "am15direct_nm"),
    ("AM15direct_eV",       "am15direct_ev"),
    ("calc_PLQY_param",     "calc_plqy_param"),
    ("choose_for_PLQY",     "choose_for_plqy"),
    ("U_energy_fit",        "u_energy_fit"),
    ("calc_Vocrad",         "calc_vocrad"),
    ("calc_Jradlim",        "calc_jradlim"),
    ("calc_Jsc",            "calc_jsc"),
    ("Tauc_plot",           "tauc_plot"),
    ("Udata_plot",          "udata_plot"),
    ("BBspectrum",          "bb_spectrum"),
    ("BBT_fit",             "bbt_fit"),
    ("load_Andor",          "load_andor"),
    ("load_Cicci",          "load_cicci"),
    ("AM15_nm",             "am15_nm"),
    ("AM15_eV",             "am15_ev"),
    ("Etr_nm",              "etr_nm"),
    ("Etr_eV",              "etr_ev"),
    ("EQE100",              "eqe100"),
    ("MMF_Eg",              "mmf_eg"),
    ("MMF_test",            "mmf_test"),
    ("MMF",                 "mmf"),
    ("PhiBB",               "phi_bb"),
    ("nm_to_eV",            "nm_to_ev"),
    ("eV_to_nm",            "ev_to_nm"),
    ("new_UE",              "new_ue"),
]

# These old names are common English words — replacements may include false
# positives in comments / strings.  Flagged in the output.
NEEDS_REVIEW = {"spectrum", "spectra", "Mobility", "Vsq"}

_COMPILED = [(re.compile(r"\b" + re.escape(old) + r"\b"), old, new)
             for old, new in REPLACEMENTS]


def apply_replacements(source: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    for pattern, old, new in _COMPILED:
        new_source, n = pattern.subn(new, source)
        if n:
            flag = " [review]" if old in NEEDS_REVIEW else ""
            changes.append(f"{old} -> {new}  ({n}x){flag}")
            source = new_source
    return source, changes


def get_source(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def set_source(cell: dict, new_source: str) -> None:
    if isinstance(cell.get("source"), list):
        # Preserve list-of-lines format with trailing newlines on each line
        # except possibly the last.
        cell["source"] = new_source.splitlines(keepends=True)
    else:
        cell["source"] = new_source


# Directories to skip — either already-updated library source, old snapshots,
# or third-party code that happens to use the same identifier names.
SKIP_DIRS = {
    # Current library source (already updated)
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-analysis-libraries-main\fte_analysis_libraries"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-analysis-libraries-main\tests"),
    # Old library snapshot — leave as-is
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\260429 FTE-analysis-libraries-main"),
    # Older copies of library modules — not user notebooks
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My modules_"),
    # Third-party packages — use "spectrum"/"spectra" as their own identifiers
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\Laboratory\pymeasure"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-impedance-main"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-impedance-main_old"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\impedance.py-main_original"),
    # Other people's code
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\old\UW\Ryan\PVtools-master"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\PV tools\SQ limit\llight_spectra\Sandy"),
}


def _is_excluded(path: Path) -> bool:
    for skip in SKIP_DIRS:
        try:
            path.relative_to(skip)
            return True
        except ValueError:
            pass
    return False


def migrate_notebook(path: Path, dry_run: bool) -> tuple[bool, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
        nb = json.loads(text)
    except Exception as exc:
        return False, [f"ERROR reading file: {exc}"]

    all_changes: list[str] = []
    modified = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = get_source(cell)
        new_source, changes = apply_replacements(source)
        if changes:
            all_changes.extend(changes)
            modified = True
            if not dry_run:
                set_source(cell, new_source)

    if modified and not dry_run:
        backup = path.with_suffix(".ipynb.bak_v091")
        backup.write_text(text, encoding="utf-8")
        path.write_text(
            json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return modified, all_changes


def migrate_pyfile(path: Path, dry_run: bool) -> tuple[bool, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, [f"ERROR reading file: {exc}"]

    new_text, changes = apply_replacements(text)
    if not changes:
        return False, []

    if not dry_run:
        backup = path.with_suffix(".py.bak_v091")
        backup.write_text(text, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")

    return True, changes


def main() -> None:
    dry_run = "--apply" not in sys.argv

    notebooks = sorted(
        p for p in NOTEBOOKS_ROOT.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts and not _is_excluded(p)
    )
    pyfiles = sorted(
        p for p in NOTEBOOKS_ROOT.rglob("*.py")
        if not _is_excluded(p) and p.name != "migrate_notebooks.py"
    )

    total = len(notebooks) + len(pyfiles)
    print(f"Scanning {len(notebooks)} notebooks and {len(pyfiles)} .py files under {NOTEBOOKS_ROOT}")
    print(f"Mode: {'DRY RUN - no files written' if dry_run else 'APPLY - writing changes + .bak_v091 backups'}\n")

    changed_files: list[tuple[Path, list[str]]] = []

    for path in notebooks:
        changed, changes = migrate_notebook(path, dry_run=dry_run)
        if changed:
            changed_files.append((path, changes))

    for path in pyfiles:
        changed, changes = migrate_pyfile(path, dry_run=dry_run)
        if changed:
            changed_files.append((path, changes))

    for path, changes in changed_files:
        rel = path.relative_to(NOTEBOOKS_ROOT)
        print(f"{'[WOULD CHANGE]' if dry_run else '[CHANGED]'}  {rel}")
        summary: dict[str, int] = {}
        for c in changes:
            summary[c] = summary.get(c, 0) + 1
        for change, count in summary.items():
            suffix = f"  (in {count} cells)" if count > 1 and path.suffix == ".ipynb" else ""
            print(f"    {change}{suffix}")
        print()

    print(f"{'Would change' if dry_run else 'Changed'} {len(changed_files)} / {total} files.")
    if dry_run and changed_files:
        print("\nRe-run with --apply to write changes (backups saved as .bak_v091).")


if __name__ == "__main__":
    main()
