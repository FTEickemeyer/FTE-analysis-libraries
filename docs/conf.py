import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'FTE Analysis Libraries'
copyright = '2026, Felix Eickemeyer'
author = 'Felix Eickemeyer'
release = '0.0.2'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'en'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'

# numpydoc
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# Suppress noisy warnings from third-party imports
nitpicky = False

suppress_warnings = [
    'ref.python',           # cross-reference ambiguities (nid/Rs/Rsh in multiple dataclasses)
    'docutils',             # rst formatting issues in pre-existing informal docstrings
]

import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='numpydoc')
