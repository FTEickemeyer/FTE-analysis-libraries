# FTE Analysis Libraries — Claude Code Context

## Project overview

Scientific Python library for photovoltaic and energy materials analysis,
developed at EPFL. Covers data import, processing, fitting, and visualization
for experiments including dye-sensitized solar cells (DSSCs), perovskite solar
cells, redox-flow batteries, and related characterization techniques (JV curves,
EIS, IPCE, PL spectroscopy, transient measurements, etc.).

The end goal of the current refactoring effort is a clean, pip-installable
package with full test coverage and auto-generated documentation.

## Package structure (target)

```
fte_analysis_libraries/      # main package (snake_case, pip-installable)
    __init__.py
    jv/                      # JV curve analysis
    eis/                     # electrochemical impedance spectroscopy
    pl/                      # photoluminescence
    transient/               # transient absorption / TCSPC
    standard/                # shared utilities (fitting, plotting, I/O)
tests/
    test_jv.py
    test_eis.py
    ...
docs/
    conf.py                  # Sphinx configuration
    index.rst
pyproject.toml               # modern packaging (replaces setup.py)
README.md
CHANGELOG.md
```

## Python environment

- Python >= 3.10
- Key dependencies: numpy, scipy, pandas, matplotlib, lmfit
- Dev dependencies: pytest, pytest-cov, sphinx, numpydoc, ruff, mypy
- Install for development: `pip install -e ".[dev]"`

## Code style

- Follow PEP 8 strictly
- Use snake_case for all functions, variables, and module names
- Use CamelCase for class names only
- All public functions and classes must have NumPy-style docstrings
- Add type hints to all function signatures
- Maximum line length: 88 characters (compatible with ruff/black)
- Prefer explicit imports over wildcard imports (`from x import *` is forbidden)

## NumPy docstring format (required for all public functions)

```python
def fit_jv_curve(voltage: np.ndarray, current: np.ndarray,
                 guess: dict | None = None) -> lmfit.ModelResult:
    """
    Fit a JV curve to extract photovoltaic parameters.

    Parameters
    ----------
    voltage : np.ndarray
        Applied voltage in volts.
    current : np.ndarray
        Measured current density in mA/cm².
    guess : dict, optional
        Initial parameter guesses. If None, auto-estimated from data.

    Returns
    -------
    lmfit.ModelResult
        Fit result containing Voc, Jsc, FF, PCE and uncertainties.

    Examples
    --------
    >>> result = fit_jv_curve(voltage, current)
    >>> print(result.params['Voc'].value)
    """
```

## Workflow rules

- ALWAYS work in a feature branch, never commit directly to main
- Branch naming: `refactor/<topic>`, `feat/<topic>`, `fix/<topic>`
- Run `ruff check .` and `pytest` before every commit
- Commit messages must follow Conventional Commits:
  - `refactor: remove dead code in eis module`
  - `feat: add type hints to jv fitting functions`
  - `fix: correct normalization in pl_analysis`
  - `docs: add NumPy docstrings to standard_functions`
- After completing a branch, open a PR — do not merge directly

## Testing rules

- Test framework: pytest
- Test files go in `/tests/`, named `test_<module>.py`
- For fitting functions: generate synthetic data with known parameters,
  fit it, and assert recovered parameters are within 1% of ground truth
- For data I/O functions: use small fixture files in `tests/fixtures/`
- Run tests with: `pytest --cov=fte_analysis_libraries --cov-report=term-missing`
- Target: 80% coverage minimum

## Linting and type checking

```bash
ruff check .          # linting
ruff format .         # formatting
mypy fte_analysis_libraries/  # type checking
pytest                # tests
```

All four must pass before a branch is considered ready to merge.

## Packaging (pip-installable from GitHub)

The package uses `pyproject.toml` (PEP 517/518). To install from GitHub:
```bash
pip install git+https://github.com/FTEickemeyer/FTE-analysis-libraries.git
# or a specific version tag:
pip install git+https://github.com/FTEickemeyer/FTE-analysis-libraries.git@v1.0.0
```

Version numbers follow Semantic Versioning: `vMAJOR.MINOR.PATCH`
- MAJOR: breaking API changes
- MINOR: new features, backward compatible
- PATCH: bug fixes

## Documentation

- Tool: Sphinx with `numpydoc` extension
- Build docs: `cd docs && make html`
- Docstrings are the single source of truth — no separate API docs to maintain

## Scientific domain notes

- Physical units should always be stated in docstrings and variable names
  (e.g., `voltage_v`, `current_density_ma_cm2`, or documented clearly)
- Fitting results should always include parameter uncertainties where possible
- Prefer `lmfit` over raw `scipy.optimize` for curve fitting (better
  uncertainty propagation and parameter bounds)
- Data import functions should return pandas DataFrames with labeled columns
