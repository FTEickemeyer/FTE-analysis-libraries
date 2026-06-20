"""
Replace weak auto-generated one-line summaries with domain-specific text.
Operates purely on the first content line inside existing docstrings.
"""
from __future__ import annotations
import re

# (function_name, old_weak_summary, new_summary)
# Use regex for the old_weak_summary so we can be flexible
FIXES: dict[str, str] = {
    # General.py
    'color_list':          'Return a list of matplotlib colours for cycling through multiple plots.',
    'round_sig':           'Round a number to a given number of significant figures.',
    'str_round_sig':       'Format a number as a string rounded to significant figures.',
    'int_arr':             'Interpolate y values onto a new x grid via scipy interp1d.',
    'interpolated_array':  'Interpolate an array onto new x positions using linear interpolation.',
    'linfit':              'Fit a linear function to data and return slope, intercept, and R².',
    'plx':                 'Print a LaTeX-formatted string to the console.',
    'v_loss':              'Calculate the voltage loss due to non-radiative recombination in eV.',
    'qfls':                'Calculate the quasi-Fermi level splitting from Voc and bandgap in eV.',
    'diff_coeff':          'Calculate the diffusion coefficient from carrier mobility (Einstein relation).',
    'mobility':            'Calculate charge-carrier mobility from the diffusion coefficient.',
    'how_long':            'Estimate and print the remaining time for a loop iteration.',
    'beep':                'Play a short audio beep to signal completion (Windows only).',
    'plot_first_n_lines':  'Plot the first n data lines from a text file for inspection.',
    'fullprint':           'Print a numpy array without truncation.',
    'is_even':             'Return True if n is even.',
    'is_odd':              'Return True if n is odd.',
    'idx_range':           'Return the index range [i_left, i_right] for a value interval in a sorted array.',
    'scattered_boxplot':   'Draw a boxplot with individual data points overlaid as a scatter.',
    'ignore_warnings':     'Decorator that suppresses Python warnings from the wrapped function.',
    'win_long_fp':         r'Prepend \\?\ to a path to enable Windows long-path support.',
    'max_len':             'Return the length of the longest sequence in a list.',
    'save_ok':             'Prompt the user before overwriting an existing file.',
    'findind':             'Return the index of the value in arr closest to the given target.',
    'df_interpolate':      'Reindex a DataFrame to a new index array using quadratic interpolation.',
    # XYdata.py
    'from_df':             'Construct an XYData object from a pandas DataFrame column.',
    'to_df':               'Export the XYData as a pandas DataFrame with labelled columns.',
    'data_check':          'Validate x/y arrays and optionally sort or trim the data.',
    'generate_empty':      'Generate an XYData with all-zero y values on a given x grid.',
    'copy':                'Return a deep copy of this object.',
    'y_of':                'Interpolate and return the y value at a given x position.',
    'normalize':           'Normalise y so that its maximum (or a chosen point) equals norm_val.',
    'quants':              'Return the (x_quantity, y_quantity) axis label strings.',
    'units':               'Return the (x_unit, y_unit) string tuple.',
    'qy_uy':               'Return the (y_quantity, y_unit) strings for the y axis.',
    'plot':                'Plot the data, with optional reference lines, insets, and fitting.',
    'plot_linfit':         'Plot data with a linear fit overlaid and residuals optionally shown.',
    'save':                'Save the data to a CSV file, prompting before overwriting.',
    'lowpass_filter':      'Apply a Butterworth low-pass filter to remove high-frequency noise.',
    'savgol':              'Smooth y data with a Savitzky-Golay filter.',
    'residual':            'Return the (weighted) residual between data and a reference array.',
    'chisquare':           'Compute the chi-square statistic between data and a model.',
    'diff':                'Compute the numerical derivative dy/dx.',
    'equidist':            'Resample y onto an equidistant x grid.',
    'cut_data_outside':    'Remove data points outside [left, right].',
    'cut_data_inside':     'Remove data points inside [left, right].',
    'remain':              'Return a copy trimmed to the interval [left, right].',
    'append':              'Append another XYData to this collection.',
    'join':                'Join two MXYData objects into one.',
    'all_values_greater_min': 'Return True if all y values exceed the given minimum.',
    'max_within':          'Return the maximum y value within [left, right].',
    'min_within':          'Return the minimum y value within [left, right].',
    'idfac_fit':           'Fit the diode ideality factor from a semi-log JV curve.',
    'polyfit':             'Fit a polynomial of specified degree to the data.',
    'load':                'Load data from a CSV or text file.',
    'load_individual':     'Load multiple individual files into a collection.',
    'save_in_one_file':    'Save all datasets to a single CSV file.',
    'save_individual':     'Save each dataset to its own CSV file.',
    # IV.py
    'det_voc':             'Determine and store the open-circuit voltage Voc in V.',
    'det_jsc':             'Determine and store the short-circuit current density Jsc in mA/cm².',
    'det_fp':              'Determine all five-parameter equivalent-circuit parameters.',
    'det_perfparam':       'Extract performance parameters (Voc, Jsc, FF, PCE, Vmpp, Jmpp).',
    'calc_jradlim':        'Calculate the radiative recombination limited current density.',
    'calc_vocrad':         'Calculate the radiative open-circuit voltage limit Voc,rad.',
    'plot_fit':            'Plot the JV curve with the five-parameter model fit overlaid.',
    'plot_ini_and_fit':    'Plot JV data with both initial guess and optimised five-parameter fit.',
    'ini_guess_rs':        'Estimate the series resistance Rs from the JV slope near Voc.',
    'ini_guess_rsh':       'Estimate the shunt resistance Rsh from the JV slope near 0 V.',
    'ini_guess_nid':       'Estimate the ideality factor from the semi-log JV curve.',
    # Spectrum.py
    'photonflux':          'Calculate the integrated photon flux of this spectrum in photons/s/cm².',
    'calc_jsc':            'Calculate Jsc from overlap of this EQE with the AM1.5G spectrum.',
    # TRPL.py
    'k1_fit':              'Fit first-order (monomolecular) recombination rate k1 to TRPL data.',
    'k1_k2_fit':           'Fit first- and second-order recombination rates k1 and k2 to TRPL data.',
    'k2_fit':              'Fit second-order (bimolecular) recombination rate k2 to TRPL data.',
    'n0_fit':              'Fit the initial carrier density n0 to TRPL data.',
    'mono_expfit':         'Fit a mono-exponential decay to extract the lifetime τ.',
    'mult2_expfit':        'Fit a bi-exponential decay to extract lifetimes τ1 and τ2.',
    'mult3_expfit':        'Fit a tri-exponential decay to extract lifetimes τ1, τ2, and τ3.',
    'mult4_expfit':        'Fit a four-exponential decay to extract four lifetimes.',
    # Electrochemistry.py
    'load_biologic_cv':    'Load a Biologic cyclic-voltammetry .mpt file.',
    'load_biologic_ca':    'Load a Biologic chronoamperometry .mpt file.',
    'load_biologic_cstc':  'Load a Biologic constant-current .mpt file.',
    # RFB.py
    'calc_conc_functions': ('Calculate vanadium species concentrations as functions of '
                            'average oxidation state.'),
    # Tkdialogs.py
    'get_filename':        'Open a file dialog and return the selected file path.',
    'get_filenames':       'Open a multi-select file dialog and return the selected paths.',
    'get_directory':       'Open a directory browser and return the selected folder path.',
}


def _replace_first_docstring_line(src: str, func_name: str, new_summary: str) -> str:
    """Find the docstring of func_name and replace its first content line."""
    # Match:  def func_name( ... ):  [maybe multi-line sig]  """  [content]
    # Strategy: find 'def func_name' followed eventually by triple-quote
    pattern = (
        r'(def\s+' + re.escape(func_name) + r'\s*\([^)]*\)[^:]*:\s*\n'
        r'(?:\s*#[^\n]*\n)*'          # optional comment lines
        r'\s*"""\s*\n)'               # opening triple-quote on its own line
        r'([^\n]+)'                   # first content line (to be replaced)
    )
    def _sub(m: re.Match) -> str:
        prefix = m.group(1)
        old_line = m.group(2)
        # Preserve leading whitespace
        indent = re.match(r'^(\s*)', old_line).group(1)
        return prefix + indent + new_summary
    new_src, n = re.subn(pattern, _sub, src, count=1)
    return new_src


def process(path: str, fixes: dict[str, str]) -> None:
    src = open(path, encoding='utf-8').read()
    changed = False
    for func_name, new_summary in fixes.items():
        new_src = _replace_first_docstring_line(src, func_name, new_summary)
        if new_src != src:
            src = new_src
            changed = True
    if changed:
        open(path, 'w', encoding='utf-8').write(src)
        print(f'  updated {path}')


if __name__ == '__main__':
    import os
    pkg = 'fte_analysis_libraries'
    for fn in ['General.py', 'XYdata.py', 'Spectrum.py', 'IV.py',
               'Electrochemistry.py', 'TRPL.py', 'RFB.py', 'PLQY.py', 'Tkdialogs.py']:
        process(os.path.join(pkg, fn), FIXES)
    print('Done.')
