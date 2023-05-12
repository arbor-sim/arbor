# The Python wrapper generated using pybind11 is a compiled dynamic library,
# with a name like _arbor.cpython-38-x86_64-linux-gnu.so
#
# The library will be installed in the same path as this file, which will imports
# the compiled part of the wrapper from the _arbor namespace.

from ._arbor import *  # noqa: F403


# Parse VERSION file for the Arbor version string.
def get_version():
    import os

    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "VERSION")) as version_file:
        return version_file.read().strip()


def modcc():
    import os, sys, subprocess

    sys.exit(
        subprocess.call(
            [os.path.join(os.path.dirname(__file__), "bin", "modcc"), *sys.argv[1:]]
        )
    )


def build_catalogue():
    import os, sys, subprocess

    sys.exit(
        subprocess.call(
            [
                os.path.join(os.path.dirname(__file__), "bin", "arbor-build-catalogue"),
                *sys.argv[1:],
            ]
        )
    )


__version__ = get_version()
__config__ = config()  # noqa:F405

# Remove get_version from arbor module.
del get_version
