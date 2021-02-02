#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
import subprocess as sp
from tempfile import TemporaryDirectory

html_static_path = ['static']

def setup(app):
    app.add_css_file('custom.css')
    app.add_object_type('generic', 'gen', 'pair: %s; generic')
    app.add_object_type('label', 'lab', 'pair: %s; label')

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.coverage'
]
source_suffix = '.rst'
master_doc = 'index'

html_logo = 'images/arbor-lines-proto-colour.svg'
html_favicon = 'images/arbor-lines-proto-colour-notext.svg'

project = 'Arbor'
copyright = '2017-2020, ETHZ & FZ Julich'
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

# Location of scripts used to generate images
this_path=os.path.split(os.path.abspath(__file__))[0]
script_path=this_path+'/scripts'
sys.path.append(script_path)

# Output path for generated images
# Dump inputs.py into tmpdir
img_path=this_path+'/gen-images'
if not os.path.exists(img_path):
    os.mkdir(img_path)

import make_images
make_images.generate(img_path)

print("-------------------------")
