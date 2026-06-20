"""
Adds 'Any' return-type annotations and 'Any' parameter annotations to every
unannotated function/method signature in the FTE Analysis Libraries,
then substitutes specific types based on parameter name patterns.

Requires: pip install libcst
Run from the project root.
"""
from __future__ import annotations
import re, sys
import libcst as cst
from libcst import parse_module, CSTTransformer, SimpleStatementLine
from libcst import Annotation, Name, Attribute, MaybeSentinel
import libcst.metadata as meta


# ── Parameter-name → annotation mapping ─────────────────────────────────────
# (checked with startswith / endswith / contains)

def _param_type(name: str, has_none_default: bool) -> str:
    """Return type annotation string for parameter `name`."""
    n = name.lower()

    # Array-like
    if n in ('x', 'y', 'z') or any(n.endswith(s) for s in ('_x', '_y', '_arr',
            '_nm', '_sf', '_pf', '_t', '_v', '_i', '_data')) or \
       any(n.startswith(s) for s in ('arr_', 'array_', 'new_', 'dat_', 'raw_')):
        base = 'np.ndarray'
        return f'{base} | None' if has_none_default else base

    if n == 'wavelengths' or n.endswith('wavelength') or n.startswith('newarr'):
        return 'np.ndarray | None' if has_none_default else 'np.ndarray'

    # String-like
    if n in ('name', 'title', 'encoding', 'kind', 'delimiter', 'plotstyle',
             'xscale', 'yscale', 'split_ch', 'used_for_fit', 'method_name',
             'name_prefix', 'subtype', 'fmt', 'mode'):
        return 'str | None' if has_none_default else 'str'
    if any(n.endswith(s) for s in ('_unit', '_name', '_path', '_file', '_dir',
                                    '_fn', '_tfn', 'filepath', 'filename',
                                    'directory')):
        return 'str | None' if has_none_default else 'str'
    if any(n.startswith(s) for s in ('filepath', 'filename', 'directory',
                                      'fn', 'tfn', 'dir_')):
        return 'str | None' if has_none_default else 'str'

    # Bool-like
    if any(n.startswith(s) for s in ('show', 'plot', 'save', 'use_', 'is_',
                                      'return_', 'both_', 'raw_data', 'reverse',
                                      'warning', 'verbose', 'uA', 'log',
                                      'take_', 'create_', 'generate_')):
        return 'bool'

    # Float-like — common physics quantities
    if any(n in s for s in (
            ('left', 'right', 'start', 'stop', 'bottom', 'top',
             'delta', 'light_int', 'cell_area', 'divisor', 'bg', 'eg',
             'temperature', 'temp', 'wavelength', 'energy', 'voltage',
             'current', 'area', 'ratio', 'factor', 'norm', 'norm_val',
             'fluence', 'voc', 'jsc', 'ff', 'pce', 'rs', 'rsh', 'nid',
             'k1', 'k2', 'n0', 'x0'))):
        return 'float | None' if has_none_default else 'float'

    # Int-like
    if n in ('header', 'n1', 'n2', 'idx', 'i', 'j', 'n', 'nr',
             'start_idx', 'stop_idx', 'y_col'):
        return 'int | None' if has_none_default else 'int'

    # Fallback
    return 'Any | None' if has_none_default else 'Any'


# ── CST Transformer ──────────────────────────────────────────────────────────

class TypeAnnotator(CSTTransformer):
    """Adds type annotations to unannotated function parameters and return types."""

    def __init__(self) -> None:
        super().__init__()
        self._lambda_depth = 0

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self._lambda_depth += 1
        return True  # visit children

    def leave_Lambda(
        self, original_node: cst.Lambda, updated_node: cst.Lambda
    ) -> cst.Lambda:
        self._lambda_depth -= 1
        return updated_node

    def leave_Param(
        self, original_node: cst.Param, updated_node: cst.Param
    ) -> cst.Param:
        # Skip lambda params (they cannot be annotated in Python)
        if self._lambda_depth > 0:
            return updated_node
        # Skip if already annotated or is self/cls/*args/**kwargs
        if updated_node.annotation is not None:
            return updated_node
        name = updated_node.name.value if isinstance(updated_node.name, cst.Name) else ''
        if name in ('self', 'cls', 'args', 'kwargs') or name.startswith('*'):
            return updated_node
        if isinstance(updated_node.star, (cst.MaybeSentinel,)):
            pass  # normal param

        # Determine if default is None
        has_none_default = False
        if updated_node.default is not None:
            d = updated_node.default
            if isinstance(d, cst.Name) and d.value == 'None':
                has_none_default = True

        type_str = _param_type(name, has_none_default)
        ann = cst.Annotation(annotation=cst.parse_expression(type_str))
        return updated_node.with_changes(annotation=ann)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if updated_node.returns is not None:
            return updated_node
        # Determine return type heuristically from the original source
        # We can't easily walk function body in CST here, so use 'Any' as default
        # (procedures will be fixed separately)
        fname = updated_node.name.value
        # Methods/functions that are clearly procedures
        if fname.startswith('_') or fname in (
                '__init__', '__repr__', '__str__', 'plot', 'save', 'show',
                'equidist', 'normalize', 'label', 'zero_data', 'qx_ux', 'qy_uy',
                'det_perfparam', 'calc_jradlim', 'calc_vocrad', 'det_fp',
                'ini_guess_rs', 'ini_guess_rsh', 'ini_guess_nid',
                'save_in_one_file', 'save_individual'):
            ret_ann = 'None'
        else:
            ret_ann = 'Any'
        ann = cst.Annotation(annotation=cst.parse_expression(ret_ann))
        return updated_node.with_changes(returns=ann)


# ── File processing ───────────────────────────────────────────────────────────

TYPING_IMPORT = 'from typing import Any\n'
NUMPY_TYPING_HINT = 'import numpy as np'  # already imported in all files

def process_file(path: str) -> None:
    print(f'  {path}', end=' ... ', flush=True)
    try:
        src = open(path, encoding='utf-8').read()
        tree = parse_module(src)
        new_tree = tree.visit(TypeAnnotator())
        new_src = new_tree.code

        # Add 'from typing import Any' if needed and not present
        if 'Any' in new_src and 'from typing import Any' not in new_src:
            if 'from typing import' in new_src:
                new_src = re.sub(
                    r'(from typing import )([^\n]+)',
                    lambda m: m.group(0) if 'Any' in m.group(2)
                              else m.group(1) + 'Any, ' + m.group(2),
                    new_src, count=1
                )
            else:
                # Insert BEFORE the first non-blank, non-comment, non-docstring
                # line that is an import statement (avoiding mid-block insertions).
                # Strategy: find the last line that starts a TOP-LEVEL import
                # (not inside a multi-line block).
                lines = new_src.splitlines(keepends=True)
                insert = 0
                paren_depth = 0
                for k, l in enumerate(lines[:60]):
                    paren_depth += l.count('(') - l.count(')')
                    if paren_depth < 0:
                        paren_depth = 0
                    # Only count as an import line if we're at depth 0
                    if paren_depth == 0 and (l.startswith('import ') or l.startswith('from ')):
                        insert = k + 1
                lines.insert(insert, TYPING_IMPORT)
                new_src = ''.join(lines)

        if new_src != src:
            open(path, 'w', encoding='utf-8').write(new_src)
            print('annotated')
        else:
            print('no changes')
    except Exception as e:
        print(f'ERROR: {e}')


if __name__ == '__main__':
    import os
    pkg = 'fte_analysis_libraries'
    for fn in ['General.py', 'XYdata.py', 'Spectrum.py', 'IV.py',
               'Electrochemistry.py', 'TRPL.py', 'RFB.py', 'PLQY.py',
               'Tkdialogs.py']:
        process_file(os.path.join(pkg, fn))
    print('Done.')
