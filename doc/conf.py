#!/usr/bin/env python3
# -*- coding: utf-8 -*-

html_static_path = ['static']

def setup(app):
    app.add_stylesheet('custom.css')
    app.add_object_type('generic', 'gen', 'pair: %s; generic')

extensions = ['sphinx.ext.todo', 'sphinx.ext.mathjax']
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
