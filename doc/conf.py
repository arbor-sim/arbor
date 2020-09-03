#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, textwrap

# Path to Python Binding (_arbor)
sys.path.insert(0, os.path.abspath('../python/arbor'))
# Path to doxygen
breathe_projects = { "Arbor": "../xml" } #exhale expects ../xml
breathe_default_project = "Arbor"
# Setup the exhale extension
exhale_args = {
    "containmentFolder":     "./cpp",
    "rootFileName":          "reference.rst",
    "rootFileTitle":         "C++ API Reference",
    "doxygenStripFromPath":  "..",
    # Suggested optional arguments
    "createTreeView":        True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "verboseBuild": False,
    "exhaleDoxygenStdin": textwrap.dedent('''
        INPUT = ../arbor
        FILE_PATTERNS = *.hpp
        EXCLUDE_PATTERNS = *impl.hpp
        EXCLUDE_SYMBOLS = ARB_DEFINE_*, ARB_PP_*
    ''')
}
cpp_id_attributes = ['__inline__', '__device__'] #CUDA declarators

html_static_path = ['static']

def setup(app):
    app.add_css_file('custom.css')
    app.add_object_type('generic', 'gen', 'pair: %s; generic')
    app.add_object_type('label', 'lab', 'pair: %s; label')

extensions = [
    'sphinx.ext.autodoc',
    'breathe',
    'exhale',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
]
source_suffix = '.rst'
master_doc = 'index'

html_logo = 'images/arbor-logo.svg'

project = 'Arbor'
copyright = '2017, ETHZ & FZ Julich'
author = 'ETHZ & FZ Julich'
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': '#dfdcdf'}

# This style makes the source code pop out a bit more
# from the background text, without being overpowering.
pygments_style = 'perldoc'

# Generate images for the documentation.
print("--- generating images ---")
import sys, os

this_path=os.path.split(os.path.abspath(__file__))[0]

# Location of scripts used to generate images
script_path=this_path+'/scripts'
sys.path.append(script_path)

# Output path for generated images
img_path=this_path+'/gen-images'
if not os.path.exists(img_path):
    os.mkdir(img_path)

import make_images
make_images.generate(img_path)

print("-------------------------")
