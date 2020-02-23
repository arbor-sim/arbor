# The Python wrapper generated using pybind11 is a compiled dynamic library,
# with a name like _arbor.cpython-38-x86_64-linux-gnu.so
#
# The library will be installed in the same path as this file, which will imports
# the compiled part of the wrapper from the _arbor namespace.

from ._arbor import *

import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
