#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def setup(app):
    app.add_stylesheet('custom.css')
    app.add_object_type('generic', 'gen', 'pair: %s; generic')

extensions = ['sphinx.ext.todo', 'sphinx.ext.mathjax']
source_suffix = '.rst'
master_doc = 'index'

project = 'Arbor'
copyright = '2017, ETHZ & FZ Julich'
author = 'ETHZ & FZ Julich'
todo_include_todos = True

html_theme = "sphinx_rtd_theme"
html_static_path = ['static']
