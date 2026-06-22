"""Patch notebooks for API fixes discovered after the v1.0.0 migration.

These renames were not caught by migrate_notebooks.py (run June 2026):

  1. import_datum → import_biologic_mpt_data  (Electrochemistry module)
  2. .add(        → .append(                  (MXYData / MTRPLData / Spectra)
     *** NEEDS REVIEW: .add() also exists on pandas objects and plain lists.
         Only replace when called on an fte_analysis_libraries collection. ***

Usage:
    python update_notebooks_post_v1.py           # dry run (shows what would change)
    python update_notebooks_post_v1.py --apply   # apply (creates .bak_post_v1 backups)
"""

import json
import re
import sys
from pathlib import Path

NOTEBOOKS_ROOT = Path(r"C:\Users\dreickem\switchdrive\Work\Python")

# Word-boundary replacements — safe to apply automatically.
WORD_REPLACEMENTS = [
    ("import_datum", "import_biologic_mpt_data"),
]

# Raw regex replacements — may have false positives; always flagged for review.
# Each entry: (compiled_pattern, replacement_string, description)
REGEX_REPLACEMENTS = [
    (
        re.compile(r"\.add\("),
        ".append(",
        ".add( -> .append(  [review: check not pandas/list .add()]",
    ),
]

SKIP_DIRS = {
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-analysis-libraries-main\fte_analysis_libraries"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-analysis-libraries-main\tests"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\260429 FTE-analysis-libraries-main"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My modules_"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\Laboratory\pymeasure"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-impedance-main"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\FTE-impedance-main_old"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\My_packages\impedance.py-main_original"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\old\UW\Ryan\PVtools-master"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\PV tools\SQ limit\llight_spectra\Sandy"),
    Path(r"C:\Users\dreickem\switchdrive\Work\Python\PV tools\IV evaluation\other\PVLIB_Python-master"),
}

_WORD_COMPILED = [
    (re.compile(r"\b" + re.escape(old) + r"\b"), old, new)
    for old, new in WORD_REPLACEMENTS
]


def apply_replacements(source: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    for pattern, old, new in _WORD_COMPILED:
        new_source, n = pattern.subn(new, source)
        if n:
            changes.append(f"{old} -> {new}  ({n}x)")
            source = new_source
    for pattern, replacement, label in REGEX_REPLACEMENTS:
        new_source, n = pattern.subn(replacement, source)
        if n:
            changes.append(f"{label}  ({n}x)")
            source = new_source
    return source, changes


def get_source(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def set_source(cell: dict, new_source: str) -> None:
    if isinstance(cell.get("source"), list):
        cell["source"] = new_source.splitlines(keepends=True)
    else:
        cell["source"] = new_source


def _is_excluded(path: Path) -> bool:
    for skip in SKIP_DIRS:
        try:
            path.relative_to(skip)
            return True
        except ValueError:
            pass
    # Skip any file inside a directory literally named "old"
    if "old" in path.parts:
        return True
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
        backup = path.with_suffix(".ipynb.bak_post_v1")
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
        backup = path.with_suffix(".py.bak_post_v1")
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
        if not _is_excluded(p) and p.name not in {"migrate_notebooks.py", "update_notebooks_post_v1.py"}
    )

    total = len(notebooks) + len(pyfiles)
    print(f"Scanning {len(notebooks)} notebooks and {len(pyfiles)} .py files under {NOTEBOOKS_ROOT}")
    print(f"Mode: {'DRY RUN - no files written' if dry_run else 'APPLY - writing changes + .bak_post_v1 backups'}\n")

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
        print("\nRe-run with --apply to write changes (backups saved as .bak_post_v1).")
    if any("[review" in c for _, changes in changed_files for c in changes):
        print("\n[!] Some changes are marked [review] — verify .add() replacements are not pandas/.list calls.")


if __name__ == "__main__":
    main()
