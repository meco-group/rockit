# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'rockit'
copyright = '2019, MECO Research Team'
author = 'MECO Research Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# napolean renders NumPy style docstrings
# viewcode add a direct link to the code in the documentation
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_gallery.gen_gallery'
]

from sphinx_gallery.sorting import ExampleTitleSortKey

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'examples',  # path where to save gallery generated examples
    'filename_pattern': '/',
    'within_subsection_order': ExampleTitleSortKey,
    'binder': {
      'org': 'https://gitlab.mech.kuleuven.be/meco-software',
      'repo': 'rockit.git',
      'branch': 'master',
      'binderhub_url': 'https://mybinder.org',
      'dependencies': ['~/.binder/requirements.txt'],
      'notebooks_dir': 'examples'
    }
}

import sphinx_gallery.binder
def patched_gen_binder_rst(fpath, binder_conf, gallery_conf):
    """Generate the RST + link for the Binder badge.
    ...
    """
    binder_conf = sphinx_gallery.binder.check_binder_conf(binder_conf)
    binder_url = sphinx_gallery.binder.gen_binder_url(fpath, binder_conf, gallery_conf)

    binder_url = binder_url.replace("/gh/","/git/")

    rst = (
            "\n"
            "  .. container:: binder-badge\n\n"
            "    .. image:: https://mybinder.org/badge_logo.svg\n"
            "      :target: {}\n"
            "      :width: 150 px\n").format(binder_url)
    return rst

sphinx_gallery.binder.gen_binder_rst = patched_gen_binder_rst

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
