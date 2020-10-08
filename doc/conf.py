#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
this_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(this_path, 'scripts')))

# Path to Python Binding (_arbor)
try:
    sys.path.append(os.path.join(os.environ['OLDPWD'],"python"))
    import arbor
except:
    autodoc_mock_imports = ['arbor._arbor']

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
copyright = '2017, ETHZ & FZ Julich'
author = 'ETHZ & FZ Julich'
todo_include_todos = True

html_theme = 'pydata_sphinx_theme'
# html_theme = "sphinx_rtd_theme"
html_context = {
    "github_user": "arbor-sim",
    "github_repo": "arbor",
    "github_version": "master",
    "doc_path": "doc",
}
html_theme_options = {
    "github_url": "https://github.com/arbor-sim/arbor",
    "use_edit_page_button": True}

# This style makes the source code pop out a bit more
# from the background text, without being overpowering.
pygments_style = 'perldoc'

# Generate images for the documentation.
print("--- generating images ---")

# Location of scripts used to generate images
import make_images

# Output path for generated images
img_path=this_path+'/gen-images'
if not os.path.exists(img_path):
    os.mkdir(img_path)

make_images.generate(img_path)

print("-------------------------")
