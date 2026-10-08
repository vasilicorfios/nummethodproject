"""Sphinx configuration for the MMA3001 numerical methods project docs."""

import sys
from pathlib import Path

# Make ``src/`` importable so autodoc can find the project modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "MMA3001 Numerical Methods Project"
author = "Project team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # parse Google-style docstrings
    "sphinx.ext.viewcode",  # link documented objects to highlighted source
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = "bysource"
# Avoid import errors when the plotting backend is unavailable on a doc build.
autodoc_mock_imports = ["matplotlib", "pandas", "openpyxl"]

html_theme = "alabaster"
