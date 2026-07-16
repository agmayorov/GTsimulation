# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path
import tomllib

sys.path.insert(0, os.path.abspath('../src'))

examples_dir = Path(__file__).parent.parent / 'examples'
symlink_target = Path(__file__).parent / 'examples' / 'notebooks'
if symlink_target.is_symlink():
    symlink_target.unlink()
symlink_target.symlink_to(examples_dir)

with open(Path(__file__).parent.parent / 'pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

project = 'GTsimulation'
copyright = '2026, GeoSpaceLab'
author = 'GeoSpaceLab'
version = pyproject['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'myst_nb',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_favicon = '_static/favicon.svg'
html_logo = '_static/gtsimulation_logo.svg'
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/agmayorov/gtsimulation",
            "icon": "fa-brands fa-github",
        },
    ],
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
}
autosectionlabel_prefix_document = True

autosummary_generate = True
autosummary_imported_members = True

# -- Options for myst_nb ------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = [
    "dollarmath",
    "html_image",
]

nb_execution_mode = "off"
myst_dmath_double_inline = True
