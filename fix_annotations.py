"""Fix wrong type annotations left by the auto-annotator."""
import re

SUBS: dict[str, list[tuple[str, str]]] = {
    'fte_analysis_libraries/XYdata.py': [
        # check_data was annotated as np.ndarray but it's a bool flag
        ('check_data: np.ndarray', 'check_data: bool'),
        # plotstyle can be dict or str or None
        ('plotstyle: str | None', 'plotstyle: Any'),
        # header can be int or 'infer' string
        ('header: int', 'header: Any'),
        # take_quants_and_units_from_file is bool, not str
        ('take_quants_and_units_from_file: str', 'take_quants_and_units_from_file: bool'),
        # plotrange is list[None] | None, not bool
        ('plotrange: bool', 'plotrange: list | None'),
        # return_data is bool, not ndarray
        ('return_data: np.ndarray', 'return_data: bool'),
        # in_name defaults to [], so it's a list
        ('in_name: str', 'in_name: Any'),
    ],
    'fte_analysis_libraries/Spectrum.py': [
        ('check_data: np.ndarray', 'check_data: bool'),
        ('plotstyle: str | None', 'plotstyle: Any'),
        # ref_PF can be string filename or ndarray
        ('ref_PF: np.ndarray', 'ref_PF: Any'),
        # showplot can be None or str or bool
        ('showplot: bool', 'showplot: Any'),
        # delta is float here
        # start/stop used in np.arange calls
    ],
    'fte_analysis_libraries/General.py': [
        # showfliers default is 'whiskers' (str), not bool
        ('showfliers: bool', 'showfliers: Any'),
        ('showmeans: bool', 'showmeans: Any'),
        ('showcaps: bool', 'showcaps: Any'),
        ('showbox: bool', 'showbox: Any'),
    ],
    'fte_analysis_libraries/IV.py': [
        ('check_data: np.ndarray', 'check_data: bool'),
        ('plotstyle: str | None', 'plotstyle: Any'),
        ('header: int', 'header: Any'),
        ('take_quants_and_units_from_file: str', 'take_quants_and_units_from_file: bool'),
        ('return_data: np.ndarray', 'return_data: bool'),
        # p0 in curve_fit context is list, not ndarray
        ('p0: np.ndarray', 'p0: Any'),
    ],
    'fte_analysis_libraries/Electrochemistry.py': [
        ('header: int', 'header: Any'),
        ('p0: np.ndarray', 'p0: Any'),
    ],
    'fte_analysis_libraries/TRPL.py': [
        ('check_data: np.ndarray', 'check_data: bool'),
        ('plotstyle: str | None', 'plotstyle: Any'),
        ('header: int', 'header: Any'),
        ('take_quants_and_units_from_file: str', 'take_quants_and_units_from_file: bool'),
        ('p0: np.ndarray', 'p0: Any'),
    ],
    'fte_analysis_libraries/RFB.py': [
        ('p0: np.ndarray', 'p0: Any'),
        ('plotstyle: str | None', 'plotstyle: Any'),
    ],
    'fte_analysis_libraries/PLQY.py': [
        ('check_data: np.ndarray', 'check_data: bool'),
    ],
}

for path, replacements in SUBS.items():
    src = open(path, encoding='utf-8').read()
    changed = False
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            print(f'  {path}: {old!r} -> {new!r}')
            changed = True
    if changed:
        open(path, 'w', encoding='utf-8').write(src)
print('Done.')
