"""Sphinx configuration for gaia_clustering."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

# Render the live example notebook from examples/ without re-executing it.
_DOCS = Path(__file__).resolve().parent
_NB_SRC = _DOCS.parent / "examples" / "example_systems.ipynb"
_NB_DST = _DOCS / "example_systems.ipynb"
if _NB_SRC.is_file():
    if _NB_DST.is_symlink() or _NB_DST.exists():
        _NB_DST.unlink()
    try:
        _NB_DST.symlink_to(os.path.relpath(_NB_SRC, _DOCS))
    except OSError:
        _NB_DST.write_bytes(_NB_SRC.read_bytes())

project = "gaia_clustering"
author = "Reinhold Willcox"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_nb",
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
    "exclude-members": "_json_default, _UNSET",
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
nb_execution_mode = "off"
nb_mime_priority_overrides = [
    ("html", "application/vnd.jupyter.widget-view+json", 10),
    ("html", "application/javascript", 20),
    ("html", "text/html", 30),
    ("html", "image/svg+xml", 40),
    ("html", "image/png", 50),
    ("html", "image/jpeg", 60),
    ("html", "text/markdown", 70),
    ("html", "text/latex", 80),
    ("html", "text/plain", 90),
]

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
    "light_css_variables": {
        "color-brand-primary": "#0f4c5c",
        "color-brand-content": "#0f4c5c",
        "color-api-name": "#0f4c5c",
        "color-sidebar-background": "#f4f7f8",
        "font-stack": "'Source Sans 3', 'Segoe UI', sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', 'Consolas', monospace",
        "font-stack--headings": "'Fraunces', Georgia, serif",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7ec8d4",
        "color-brand-content": "#7ec8d4",
        "color-api-name": "#7ec8d4",
        "color-sidebar-background": "#121a1c",
        "font-stack": "'Source Sans 3', 'Segoe UI', sans-serif",
        "font-stack--monospace": "'IBM Plex Mono', 'Consolas', monospace",
        "font-stack--headings": "'Fraunces', Georgia, serif",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = None

nitpicky = False
