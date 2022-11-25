#!/bin/bash
# A script that can be ran in the PyPA manywheel containers if you want to produce uploadable wheels for PyPI.
# Steps:
# 1. Prepare a (temporary) working directory (referred to as $LOCAL_WORK_DIR). 
# 2. Have the version of Arbor you want to build manylinux compliant wheels for available at $LOCAL_WORK_DIR/arbor
# 3. Start an instance of the docker image with $LOCAL_WORK_DIR mounted at /src_dir
#    Then, run /src_dir/arbor/scripts/build-wheels.sh
#    Using podman, the follow command can be used:
#    podman run -v $LOCAL_WORK_DIR:/src_dir:Z -ti quay.io/pypa/manylinux2014_x86_64 /src_dir/arbor/scripts/build-wheels.sh
# 4. After the run is complete, find in $LOCAL_WORK_DIR/wheelhouse the wheels ready for PyPI.
#    $LOCAL_WORK_DIR/builtwheel contains the wheel before auditwheel has processed them. Can be discarded,
#    or used for analysis in case of build failure.

set -e -u -x

rm -rf /src_dir/arbor/_skbuild
rm -rf /opt/python/cp36-cp36m # Python build toolchain does not work on Py3.6

export CIBUILDWHEEL=1 #Set condition for cmake

for PYBIN in /opt/python/cp3*/bin; do
    "${PYBIN}/python" -m pip install pip auditwheel -U
    export PATH="${PYBIN}":$PATH
    "${PYBIN}/python" -m pip wheel --wheel-dir="/src_dir/builtwheel${PYBIN}/" /src_dir/arbor
    "${PYBIN}/python" -m auditwheel repair /src_dir/builtwheel${PYBIN}/arbor*.whl -w /src_dir/wheelhouse
done

/opt/python/cp310-cp310/bin/python /src_dir/arbor/scripts/patchwheel.py /src_dir/wheelhouse

# Todo: Install packages and test
