"""
Generate NumPy-style docstrings for public functions and classes that lack them.

Adds:
  - one-line summary (derived from function/class name)
  - Parameters section (from type annotations, with physical units)
  - Returns section (from return-type annotation)
  - short Examples section

Run from the project root:  python gen_docstrings.py
"""
from __future__ import annotations
import re
import libcst as cst
from libcst import parse_module, CSTTransformer


# ── Name-to-English conversion ───────────────────────────────────────────────

_VERB = {
    'load': 'Load', 'save': 'Save', 'plot': 'Plot', 'calc': 'Calculate',
    'calculate': 'Calculate', 'det': 'Determine', 'determine': 'Determine',
    'gen': 'Generate', 'generate': 'Generate', 'get': 'Get', 'set': 'Set',
    'find': 'Find', 'fit': 'Fit', 'add': 'Add', 'remove': 'Remove',
    'show': 'Show', 'check': 'Check', 'read': 'Read', 'write': 'Write',
    'convert': 'Convert', 'normalize': 'Normalize', 'integrate': 'Integrate',
    'interpolate': 'Interpolate', 'ini': 'Initialize', 'init': 'Initialize',
    'update': 'Update', 'extract': 'Extract', 'compute': 'Compute',
    'run': 'Run', 'upload': 'Upload', 'import': 'Import', 'export': 'Export',
    'create': 'Create', 'build': 'Build', 'make': 'Make',
    'estimate': 'Estimate', 'guess': 'Estimate initial', 'batch': 'Batch-process',
    'split': 'Split', 'merge': 'Merge', 'combine': 'Combine',
    'append': 'Append', 'apply': 'Apply', 'copy': 'Copy',
    'cut': 'Cut', 'filter': 'Filter', 'scale': 'Scale',
    'draw': 'Draw', 'print': 'Print', 'label': 'Label',
    'reset': 'Reset', 'clear': 'Clear', 'sort': 'Sort',
}

_NOUN = {
    # photovoltaics
    'Voc': 'open-circuit voltage', 'voc': 'open-circuit voltage',
    'Jsc': 'short-circuit current density', 'jsc': 'short-circuit current density',
    'FF': 'fill factor', 'ff': 'fill factor',
    'PCE': 'power conversion efficiency', 'pce': 'power conversion efficiency',
    'EQE': 'external quantum efficiency', 'eqe': 'external quantum efficiency',
    'IQE': 'internal quantum efficiency',
    'SQ': 'Shockley-Queisser', 'sq': 'Shockley-Queisser',
    'nid': 'ideality factor', 'Nid': 'ideality factor',
    'Rs': 'series resistance', 'Rsh': 'shunt resistance',
    'rs': 'series resistance', 'rsh': 'shunt resistance',
    # optics / spectroscopy
    'PL': 'photoluminescence', 'pl': 'photoluminescence',
    'PLQY': 'photoluminescence quantum yield',
    'TRPL': 'time-resolved photoluminescence',
    'PF': 'photon flux', 'SF': 'spectral flux',
    'BBT': 'black-body temperature', 'bbt': 'black-body temperature',
    'spec': 'spectrum', 'Spec': 'spectrum', 'spectra': 'spectra',
    'abs': 'absorbance', 'Abs': 'absorbance',
    'diff': 'differential', 'Diff': 'differential',
    'EL': 'electroluminescence',
    'AM15': 'AM1.5G spectrum',
    # electrochemistry
    'CV': 'cyclic voltammetry', 'cv': 'cyclic voltammetry',
    'CA': 'chronoamperometry', 'ca': 'chronoamperometry',
    'EIS': 'electrochemical impedance spectroscopy',
    'CstC': 'constant-current', 'cstc': 'constant-current',
    'OCV': 'open-circuit voltage',
    # redox-flow battery
    'RFB': 'redox-flow battery', 'rfb': 'redox-flow battery',
    'SOC': 'state of charge', 'soc': 'state of charge',
    'conc': 'concentration', 'Conc': 'concentration',
    'V2': 'vanadium(II)', 'V3': 'vanadium(III)', 'V4': 'vanadium(IV)',
    'V5': 'vanadium(V)',
    # data / general
    'df': 'DataFrame', 'Df': 'DataFrame',
    'arr': 'array', 'Arr': 'array',
    'mpt': 'Biologic MPT file',
    'IV': 'current-voltage', 'JV': 'current density-voltage',
    'param': 'parameters', 'Param': 'parameters',
    'perf': 'performance', 'Perf': 'performance',
    'fp': 'five-parameter', 'FP': 'five-parameter',
    'bkg': 'background', 'Bkg': 'background', 'bg': 'background',
    'norm': 'normalised', 'Norm': 'normalised',
    'linfit': 'linear fit', 'expfit': 'exponential fit',
    'idfac': 'ideality factor',
    'data': 'data', 'Data': 'data',
}

_DUNDER = {
    'init': 'Initialize the object.',
    'repr': 'Return a string representation.',
    'str': 'Return a string representation.',
    'mul': 'Multiply element-wise with another object or scalar.',
    'add': 'Add another object or scalar element-wise.',
    'sub': 'Subtract another object or scalar element-wise.',
    'truediv': 'Divide element-wise by another object or scalar.',
    'len': 'Return the number of elements.',
    'iter': 'Iterate over elements.',
    'getitem': 'Return element at the given index.',
    'setitem': 'Set element at the given index.',
    'contains': 'Check membership.',
    'call': 'Call the object.',
    'enter': 'Enter the context manager.',
    'exit': 'Exit the context manager.',
}

_CLASS_DESC = {
    'XYData': 'Container for a single (x, y) data set with units and plotting helpers.',
    'MXYData': 'Container for a collection of XYData objects (multi-spectrum).',
    'IVData': 'Current-voltage (JV) curve with parameter extraction and fitting.',
    'FiveParam': 'Five-parameter solar-cell equivalent-circuit model dataclass.',
    'PerfData': 'Solar-cell performance parameters dataclass (Voc, Jsc, FF, PCE, …).',
    'TRPLData': 'Single time-resolved photoluminescence decay trace.',
    'MTRPLData': 'Collection of TRPLData objects for batch TRPL analysis.',
    'ExpParam': 'Experimental parameters for a PLQY measurement run.',
    'PLQYDataset': 'Dataset container for absolute PLQY calculation.',
}


def _name_to_sentence(name: str) -> str:
    """Convert a snake_case or CamelCase name to a readable summary sentence."""
    # dunder
    if name.startswith('__') and name.endswith('__'):
        key = name[2:-2]
        return _DUNDER.get(key, f'Special method ``{name}``.')

    bare = name.lstrip('_')
    # Split on underscores; also split on CamelCase boundaries inside tokens
    tokens: list[str] = []
    for chunk in bare.split('_'):
        # sub-split CamelCase: 'loadBiologic' → ['load', 'Biologic']
        sub = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+', chunk)
        tokens.extend(sub if sub else [chunk])

    if not tokens:
        return name + '.'

    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            parts.append(_VERB.get(tok, _VERB.get(tok.lower(), tok.capitalize())))
        else:
            mapped = _NOUN.get(tok) or _NOUN.get(tok.lower()) or tok
            parts.append(mapped)

    sentence = ' '.join(parts)
    if not sentence.endswith('.'):
        sentence += '.'
    return sentence[0].upper() + sentence[1:]


# ── Physical unit inference ───────────────────────────────────────────────────

_UNIT_PATTERNS: list[tuple[str, str]] = [
    (r'\bvoltage\b|\bvoc\b|\bvmpp\b|\bv_oc\b|\bvmax\b',        'in V'),
    (r'\bjsc\b|\bjmpp\b|\bcurrent_density\b|\bcurrent\b',        'in mA/cm²'),
    (r'\bwavelength\b|_nm\b|arr_nm\b|nm_',                       'in nm'),
    (r'\benergy\b|_ev\b|arr_ev\b|energy_',                       'in eV'),
    (r'\btime\b|arr_t\b|_t\b(?!emp)|lifetime\b|\btau\b',        'in ns'),
    (r'\barea\b|cell_area\b',                                     'in cm²'),
    (r'\btemperature\b|\btemp\b',                                 'in K'),
    (r'\brs\b|series_res',                                        'in Ω·cm²'),
    (r'\brsh\b|shunt_res',                                        'in Ω·cm²'),
    (r'\blight_int\b|\birradiance\b|\bphoton_flux\b',            'in mW/cm²'),
    (r'\bfreq\b|\bfrequency\b',                                   'in Hz'),
    (r'\bpce\b',                                                  'in %'),
    (r'\bconc\b|concentration\b',                                 'in mol/L'),
    (r'\bcapacity\b|cap\b',                                       'in mAh'),
    (r'\bsoc\b|state_of_charge\b',                               'as fraction (0–1)'),
    (r'\bcurrent_a\b|\bcurrent_ma\b',                            'in mA'),
    (r'\bfluence\b|\bdose\b',                                     'in photons/cm²'),
    (r'\bkb\b|boltzmann\b',                                      'in eV/K'),
]


def _unit_suffix(param_name: str) -> str:
    n = param_name.lower()
    for pattern, unit in _UNIT_PATTERNS:
        if re.search(pattern, n):
            return f', {unit}'
    return ''


# ── Docstring text builder ────────────────────────────────────────────────────

def _param_doc(name: str, type_str: str) -> tuple[str, str]:
    """Return (type_line, description_line) for one parameter."""
    unit = _unit_suffix(name)
    desc = name.replace('_', ' ')
    if unit:
        desc += unit
    desc = desc.capitalize() + '.'
    return f'{name} : {type_str}', f'    {desc}'


def _return_doc(ret: str) -> tuple[str, str]:
    """Return (type_line, description_line) for the return value."""
    desc_map = {
        'np.ndarray': 'Result array.',
        'pd.DataFrame': 'Result table.',
        'float': 'Computed value.',
        'int': 'Computed integer value.',
        'bool': 'Boolean result.',
        'str': 'String result.',
        'Any': 'Computed result.',
        'None': '',
        'tuple[Any, Any]': 'Pair of results.',
    }
    desc = desc_map.get(ret, 'Computed result.')
    return ret, f'    {desc}'


def _build_docstring(
    fname: str,
    params: list[tuple[str, str]],   # [(name, type_str), ...]
    ret: str | None,
    class_name: str | None,
    indent: str,
) -> str:
    """Assemble a NumPy-style docstring with given indent."""
    i = indent        # content indent (e.g. 8 spaces for a method)
    lines: list[str] = []

    summary = _name_to_sentence(fname)
    lines.append(f'{i}{summary}')

    # Filter boring params
    skip = {'self', 'cls', 'args', 'kwargs'}
    real = [(n, t) for n, t in params if n not in skip and not n.startswith('**') and not n.startswith('*')]

    if real:
        lines += [f'{i}', f'{i}Parameters', f'{i}----------']
        for name, type_str in real:
            tl, dl = _param_doc(name, type_str)
            lines.append(f'{i}{tl}')
            lines.append(f'{i}{dl}')

    if ret and ret not in ('None', 'none'):
        tl, dl = _return_doc(ret)
        lines += [f'{i}', f'{i}Returns', f'{i}-------', f'{i}{tl}']
        if dl.strip():
            lines.append(f'{i}{dl}')

    # Example
    lines.append(f'{i}')
    lines.append(f'{i}Examples')
    lines.append(f'{i}--------')
    if class_name:
        lines.append(f'{i}>>> obj.{fname}()')
    else:
        lines.append(f'{i}>>> {fname}()')

    body = '\n'.join(lines)
    return f'"""\n{body}\n{indent}"""'


# ── libcst transformer ────────────────────────────────────────────────────────

def _has_docstring(body: cst.BaseSuite) -> bool:
    """Return True if the first statement in body is a string literal."""
    if not isinstance(body, cst.IndentedBlock):
        return False
    stmts = body.body
    if not stmts:
        return False
    first = stmts[0]
    if not isinstance(first, cst.SimpleStatementLine):
        return False
    if not first.body:
        return False
    expr = first.body[0]
    if not isinstance(expr, cst.Expr):
        return False
    val = expr.value
    return isinstance(val, (cst.SimpleString, cst.ConcatenatedString,
                             cst.FormattedString))


def _extract_params(params: cst.Parameters) -> list[tuple[str, str]]:
    """Extract (name, type_str) from a Parameters node."""
    result: list[tuple[str, str]] = []
    for p in (*params.params, *(params.kwonly_params or [])):
        if not isinstance(p, cst.Param):
            continue
        name = p.name.value if isinstance(p.name, cst.Name) else str(p.name)
        if p.annotation:
            try:
                type_str = cst.parse_module('').code_for_node(p.annotation.annotation)
            except Exception:
                type_str = 'Any'
        else:
            type_str = 'Any'
        result.append((name, type_str))
    return result


def _get_return(node: cst.FunctionDef) -> str | None:
    if node.returns is None:
        return None
    try:
        return cst.parse_module('').code_for_node(node.returns.annotation)
    except Exception:
        return 'Any'


class DocstringAdder(CSTTransformer):
    """Insert NumPy docstrings into functions and classes that lack them."""

    def __init__(self) -> None:
        super().__init__()
        self._class_stack: list[str] = []

    # ── class tracking ────────────────────────────────────────────────────────

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._class_stack.append(node.name.value)
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        name = self._class_stack.pop() if self._class_stack else updated_node.name.value
        if _has_docstring(updated_node.body):
            return updated_node
        indent = '    ' * (len(self._class_stack) + 1)
        desc = _CLASS_DESC.get(name, f'Container class for {name} data and operations.')
        ds_text = f'"""\n{indent}{desc}\n{indent}"""'
        return updated_node.with_changes(
            body=self._prepend(updated_node.body, ds_text)
        )

    # ── function / method ─────────────────────────────────────────────────────

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if _has_docstring(updated_node.body):
            return updated_node

        fname = updated_node.name.value
        # Skip private dunder except __init__
        if fname.startswith('__') and fname not in ('__init__', '__repr__',
                '__str__', '__mul__', '__add__', '__sub__', '__truediv__',
                '__len__', '__iter__', '__getitem__'):
            return updated_node

        params = _extract_params(updated_node.params)
        ret = _get_return(updated_node)
        class_name = self._class_stack[-1] if self._class_stack else None

        # indent: class_depth levels + 1 for function body
        indent = '    ' * (len(self._class_stack) + 1)

        ds_text = _build_docstring(fname, params, ret, class_name, indent)
        return updated_node.with_changes(
            body=self._prepend(updated_node.body, ds_text)
        )

    # ── helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _prepend(body: cst.BaseSuite, ds_text: str) -> cst.BaseSuite:
        """Insert a docstring as the first statement of an IndentedBlock."""
        if not isinstance(body, cst.IndentedBlock):
            return body
        ds_node = cst.SimpleStatementLine(
            body=[cst.Expr(value=cst.SimpleString(ds_text))],
            leading_lines=[],
        )
        new_body = (ds_node, *body.body)
        return body.with_changes(body=new_body)


# ── File processing ───────────────────────────────────────────────────────────

def process_file(path: str) -> None:
    print(f'  {path}', end=' ... ', flush=True)
    try:
        src = open(path, encoding='utf-8').read()
        tree = parse_module(src)
        new_tree = tree.visit(DocstringAdder())
        new_src = new_tree.code
        if new_src != src:
            open(path, 'w', encoding='utf-8').write(new_src)
            print('updated')
        else:
            print('no changes')
    except Exception as e:
        import traceback
        print(f'ERROR: {e}')
        traceback.print_exc()


if __name__ == '__main__':
    import os
    pkg = 'fte_analysis_libraries'
    for fn in [
        'General.py', 'XYdata.py', 'Spectrum.py', 'IV.py',
        'Electrochemistry.py', 'TRPL.py', 'RFB.py', 'PLQY.py', 'Tkdialogs.py',
    ]:
        process_file(os.path.join(pkg, fn))
    print('Done.')
