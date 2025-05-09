# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../benchmarks"))
sys.path.insert(0, os.path.abspath(".."))

sys.path.append("_ext")


# -- Project information -----------------------------------------------------

project = "arkouda"
copyright = "2020, Michael Merrill and William Reus"
author = "Michael Merrill and William Reus"

# The full version, including alpha/beta/rc tags
# release = _version.get_versions()["version"]
release = ""


# -- General configuration ---------------------------------------------------
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    "autoapi.extension",
    "ak_sphinx_extensions",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
]


mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"


# Include __init__.pyi files
autodoc_mock_imports = ["arkouda.numpy", "arkouda.pandas", "arkouda.scipy"]

source_suffix = [".rst", ".md"]

myst_enable_extensions = ["deflist", "linkify"]


# path to directory containing files to autogenerate docs from comments
autoapi_dirs = ["../arkouda"]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autoapi_member_order = "alphabetical"

# do not create api reference for
autoapi_ignore = [
    "*migrations*",
    "*_version.py",
    "*decorators.py",
    "*message.py",
    "*arkouda/numpy/dtypes.py",
    "*arkouda/numpy/imports/*",
    "*arkouda/dataframe.py",
    "*arkouda/groupbyclass.py",
    "*arkouda/series.py",
    "*arkouda/scipy/*py",
    "*arkouda/scipy/stats/*py",
    "*arkouda/scipy/special/*py",
    #   Do not include the old module locations to reduce cross-reference WARNINGS
    "*arkouda/dtypes/*",
    "*arkouda/numeric/*",
    "*arkouda/pdarrayclass/*",
    "*arkouda/pdarraycreation/*",
    "*arkouda/pdarraymanipulation/*",
    "*arkouda/pdarraysetops/*",
    "*arkouda/random/*",
    "*arkouda/segarray/*",
    "*arkouda/sorting/*",
    "*arkouda/strings/*",
    "*arkouda/timeclass/*",
    "*arkouda/util/*",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add release substitution variable
substitutions = [("|release|", release)]
