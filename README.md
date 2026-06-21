# FTE Analysis Libraries

Scientific Python library for photovoltaic and energy materials analysis (EPFL / FTE group). Covers data import, processing, fitting, and visualization for JV curves, EIS, PL spectroscopy, TRPL, PLQY, redox-flow batteries, and related techniques.

## Installation

**From GitHub (latest):**
```bash
pip install git+https://github.com/FTEickemeyer/FTE-analysis-libraries.git
```

**From GitHub (specific version tag):**
```bash
pip install git+https://github.com/FTEickemeyer/FTE-analysis-libraries.git@v0.0.2
```

**Local development install (from the cloned directory):**
```bash
pip install -e .
```

## Quick usage example

```python
import numpy as np
from fte_analysis_libraries.IV import IVData

# Create a synthetic JV curve from five-parameter model
V = np.linspace(-0.05, 1.1, 200)
iv = IVData.from_j0(V, J0=1e-12, Jph=20e-3, nid=1.5, Rs=5.0,
                     Rsh=1000.0, light_int=100.0)

# Extract Voc, Jsc, FF, PCE
iv.fit_param()
print(f'Voc = {iv.Voc:.3f} V,  Jsc = {iv.Jsc:.2f} mA/cm²')
print(f'FF  = {iv.FF:.1f}%,    PCE = {iv.PCE:.1f}%')

# Plot
iv.plot(title='Synthetic JV curve')
```

## Modules

| Module | Purpose |
|---|---|
| `General` | Physical constants, math helpers, plotting utilities |
| `XYdata` | Base `XYData`/`MXYData` classes — arrays with units, plot/save methods |
| `Spectrum` | Optical spectra: Andor/OceanOptics loaders, BBT fit, EQE, PLQY prep |
| `IV` | JV curve analysis: Voc/Jsc/FF/PCE, loss analysis, SQ-limit calculations |
| `Electrochemistry` | CV and EIS loading (Biologic .mpt); equivalent-circuit fitting |
| `PLQY` | Absolute PLQY from integrating-sphere spectra |
| `TRPL` | Time-resolved PL: multi-exponential fits, IRF deconvolution, drift-diffusion |
| `RFB` | Redox-flow battery: vanadium concentrations, cycling analysis, efficiencies |
| `Tkdialogs` | Tkinter file/directory dialog helpers |

## Documentation

API documentation (Sphinx): build locally with

```bash
cd docs && make html
```

then open `docs/_build/html/index.html`.
