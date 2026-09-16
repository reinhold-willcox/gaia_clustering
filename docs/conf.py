"""Sphinx configuration for gaia_clustering."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "gaia_clustering"
author = "Reinhold Willcox"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "exclude-members": "_json_default",
}
# Keep the HTML build light: these packages are imported at module level
# but are not required to render signatures and docstrings.
autodoc_mock_imports = [
    "pymc",
    "pytensor",
    "pytensor.tensor",
    "arviz",
    "plotly",
    "plotly.graph_objects",
    "corner",
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "gaia_clustering"
html_theme_options = {
    "source_repository": "https://github.com/reinhold-willcox/gaia_clustering/",
    "source_branch": "main",
    "source_directory": "docs/",
}
html_static_path = ["_static"]

nitpicky = False
