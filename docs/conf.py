# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src'))

examples_dir = Path(__file__).parent.parent / 'examples'
symlink_target = Path(__file__).parent / 'examples' / 'notebooks'
if symlink_target.is_symlink():
    symlink_target.unlink()
symlink_target.symlink_to(examples_dir)

project = 'GTsimulation'
copyright = '2026, SpaceLab'
author = 'SpaceLab'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.apidoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/favicon.svg'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}

autosummary_generate = True
autosectionlabel_prefix_document = True

# -- Options for apidoc ------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/apidoc.html#configuration

apidoc_modules = [
    {
        'path': '../src/gtsimulation',
        'destination': 'reference/',
        'no_headings': True,
    },
]

# -- Options for myst_nb ------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "dollarmath",
]

nb_execution_mode = "off"
myst_dmath_double_inline = True
