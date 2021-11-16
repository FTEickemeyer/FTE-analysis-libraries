# Analysis Libraries

Installation:
1. save the content of this directory in a new directory "FTE-analysis-libraries-main" on your computer
2. open a command prompt
3. in the prompt, go to the directory in which "FTE-analysis-libraries-main" is in (via cd ...)
4. type: python -m pip install FTE-analysis-libraries-main

An assortment of analysis libraries.

## Components

**Use**

To `import` the modules use the form
```python
from FTE_analysis_libraries[.<component>] import <module>
# or
import FTE_analysis_libraries[.<component>].<module>

examples where every object is imported:

from FTE_analysis_libraries.General import *

import FTE_analysis_libraries.Spectrum as sm
sm.calc_laser_power(660, bg_eV = f1240/800, A = 1e-6, details = True)

```
where `<component>` is the name of the component (if needed) and `<module>` is the name of the module. Any modules in the Standard Component do not require a component name, while modules in all other components do.
