#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Add /scripts to path. Used for Furo theme and to generate images
this_path = os.path.split(os.path.abspath(__file__))[0]
script_path = this_path + "/scripts"
sys.path.append(script_path)

html_static_path = ["static"]
html_css_files = ["htmldiag.css"]
html_js_files = ["domarrow.js", "latest-warning.js"]


def setup(app):
    app.add_object_type("generic", "gen", "pair: %s; generic")
    app.add_object_type("label", "lab", "pair: %s; label")


extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
]
source_suffix = ".rst"
master_doc = "index"

html_favicon = "images/arbor-lines-proto-colour-notext.svg"

intersphinx_mapping = {
    "lfpykit": ("https://lfpykit.readthedocs.io/en/latest/", None),
}

project = "Arbor"
copyright = "2017-2025, ETHZ & FZJ"
author = "ETHZ & FZJ"
todo_include_todos = True

html_theme = "furo"
html_theme_options = {
    "dark_logo": "arbor-lines-proto-colour-dark.svg",
    "light_logo": "arbor-lines-proto-colour.svg",
}

# This style makes the source code pop out a bit more
# from the background text, without being overpowering.
pygments_style = "perldoc"

# Generate images for the documentation.
print("--- generating images ---")

# Output path for generated images
# Dump inputs.py into tmpdir
img_path = this_path + "/gen-images"
if not os.path.exists(img_path):
    os.mkdir(img_path)

import make_images  # noqa:E402

make_images.generate(img_path)

print("-------------------------")
