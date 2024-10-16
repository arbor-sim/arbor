# NOTE this is a placeholder until we make pybind11 bindings for modcc
def cli():
    import os
    import sys
    import subprocess

    sys.exit(
        subprocess.call(
            [os.path.join(os.path.dirname(__file__), "bin", "modcc"), *sys.argv[1:]]
        )
    )
