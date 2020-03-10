#!/usr/bin/env python3
# -*- coding: utf-8 -*-


html_static_path = ['static']

def setup(app):
    app.add_stylesheet('custom.css')
    app.add_object_type('generic', 'gen', 'pair: %s; generic')

extensions = [
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

# Create images.
import sample_tree
sample_tree.generate(img_path)

import morphology
morphology.generate(img_path)

print("-------------------------")
