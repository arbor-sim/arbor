#!/bin/bash
# A script that can be ran in the PyPA manywheel containers if you want to produce uploadable wheels for PyPI.
# Steps:
# 1. Prepare a (temporary) working directory (referred to as $LOCAL_WORK_DIR). 
# 2. Have the version of Arbor available at $LOCAL_WORK_DIR/arbor
# 3. Start an instance of the docker image with $LOCAL_WORK_DIR mounted at /src_dir
#    Using podman, the follow command would suffice. as follows:
#    podman run -v $LOCAL_WORK_DIR:/src_dir:Z -ti quay.io/pypa/manylinux2014_x86_64 /src_dir/arbor/scripts/build-wheels.sh
# 4. On the prompt in the machine, execute this script.
#    /src_dir/arbor/scripts/build-wheels.sh
# 5. After the run is complete, find in $LOCAL_WORK_DIR/wheelhouse the wheels ready for PyPI.

set -e -u -x

yum -y install libxml2-devel libxml2-static
/opt/python/cp39-cp39/bin/pip install ninja cmake
rm -rf /opt/python/cp35-cp35m #skip building for Python 3.5

rm -rf /src_dir/arbor/_skbuild

export CIBUILDWHEEL=1 #Set condition for cmake

for PYBIN in /opt/python/cp*/bin; do
    "${PYBIN}/python" -m pip install wheel scikit-build auditwheel
    export PATH="${PYBIN}":$PATH
    "${PYBIN}/python" -m pip wheel --wheel-dir="/src_dir/builtwheel${PYBIN}/" /src_dir/arbor
    "${PYBIN}/python" -m auditwheel repair /src_dir/builtwheel${PYBIN}/arbor*.whl -w /src_dir/wheelhouse
done

# Todo: Install packages and test
