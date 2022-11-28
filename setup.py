from sys import executable as python
from skbuild import setup

# Hard coded options, because scikit-build does not do build options.
# Override by instructing CMAKE, e.g.:
# pip install . -- -DARB_USE_BUNDLED_LIBS=ON -DARB_WITH_MPI=ON -DARB_GPU=cuda
with_mpi = False
with_gpu = "none"
with_vec = False
arch = "none"
use_libs = True
build_type = "Release"  # this is ok even for debugging, as we always produce info

setup(
    cmake_args=[
        "-DARB_WITH_PYTHON=on",
        f"-DPYTHON_EXECUTABLE={python}",
        f"-DARB_WITH_MPI={with_mpi}",
        f"-DARB_VECTORIZE={with_vec}",
        f"-DARB_ARCH={arch}",
        f"-DARB_GPU={with_gpu}",
        f"-DARB_USE_BUNDLED_LIBS={use_libs}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ],
)
