RED='\033[0;31m'
YELLOW='\033[0;33m'
CLEAR='\033[0m'

error()    {>&2 echo -e "${RED}ERROR${CLEAR}: $1"; exit 1;}
progress() { echo; echo -e "${YELLOW}STATUS${CLEAR}: $1"; echo;}

base_path=`pwd`
build_path=build-${BUILD_NAME}

#
# print build-specific and useful information
#
progress "build environment"

compiler_version=`${CXX} --version | grep -m1 ""`
cmake_version=`cmake --version | grep version | awk '{print $3}'`

echo "compiler   : ${compiler_version}"
echo "cmake      : ${cmake_version}"
echo "build path : ${build_path}"
echo "base path  : ${base_path}"
echo "python3    : $(which python3)"
echo "python3ver : $(python3 --version)"

if [[ "${WITH_DISTRIBUTED}" == "mpi" ]]; then
    echo "mpi        : on"
    export OMPI_CC=${CC}
    export OMPI_CXX=${CXX}
    CC="mpicc"
    CXX="mpicxx"
    launch="mpiexec -n 4"
    # on mac:
    # --oversubscribe flag allows more processes on a node than processing elements
    # --mca btl tcp,self for Open MPI to use the "tcp" and "self" Byte Transfer Layers for transporting MPI messages
    # "self" to deliver messages to the same rank as the sender
    # "tcp" sends messages across TCP-based networks (Transmission Control Protocol with Internet Protocol)
    if [[ "$TRAVIS_OS_NAME" = "osx" ]]; then
        launch="${launch} --oversubscribe --mca btl tcp,self"
    fi
    WITH_MPI="ON"
else
    echo "mpi        : off"
    launch=""
    WITH_MPI="OFF"
fi

if [[ "${WITH_PYTHON}" == "true" ]]; then
    echo "python     : on"
    ARB_WITH_PYTHON="ON"
    # The build process creates the arbor module in build_path/python/arbor
    export PYTHONPATH=$PYTHONPATH:${base_path}/${build_path}/python
    python_path=$base_path/python
    echo "python src : ${python_path}"
    echo "PYTHONPATH : ${PYTHONPATH}"

    if [[ "$TRAVIS_OS_NAME" = "linux" ]]; then
        pypref=$(python-config --prefix)
        PY_FLAGS="-DPYTHON_EXECUTABLE=$pypref/bin/python3.6 -DPYTHON_INCLUDE_DIR=$pypref/include/python3.6m -DPYTHON_LIBRARY=$pypref/lib/python3.6/config-3.6m-x86_64-linux-gnu/libpython3.6m.so"
    else
        PY_FLAGS=""
    fi
else
    echo "python     : off"
    ARB_WITH_PYTHON="OFF"
fi

#
# make build path
#
mkdir -p $build_path
cd $build_path

#
# run cmake
#
progress "Configuring with cmake"

# Fix CMake/Homebrew/XCode mess. See: https://github.com/apple/swift/pull/32436
if which xcrun >/dev/null; then
    typeset -x CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}":$(xcrun --sdk macosx --show-sdk-path)/usr
fi

cmake_flags="-DARB_WITH_ASSERTIONS=ON -DARB_WITH_NEUROML=${WITH_NEUROML} -DARB_WITH_MPI=${WITH_MPI} -DARB_WITH_PYTHON=${ARB_WITH_PYTHON} -DARB_ARCH=${ARCH} ${CXX_FLAGS} ${PY_FLAGS}"
echo "cmake flags: ${cmake_flags}"
cmake .. ${cmake_flags} || error "unable to configure cmake"

export ARB_NUM_THREADS=2

progress "C++ unit tests"
make unit -j4                || error "building unit tests"
./bin/unit --gtest_color=no  || error "running unit tests"

progress "C++ modcc unit testss"
make unit-modcc -j4                || error "building modcc unit tests"
./bin/unit-modcc --gtest_color=no  || error "running modcc unit tests"

progress "C++ distributed unit tests (local)"
make unit-local -j4          || error "building local distributed unit tests"
./bin/unit-local             || error "running local distributed unit tests"

if [[ "${WITH_DISTRIBUTED}" == "mpi" ]]; then
    progress "C++ distributed unit tests (MPI)"
    make unit-mpi -j4        || error "building MPI distributed unit tests"
    ${launch} ./bin/unit-mpi || error "running MPI distributed unit tests"
fi

if [[ "${WITH_PYTHON}" == "true" ]]; then
    progress "Building python module"
    make pyarb -j4                                                                                || error "building pyarb"
    progress "Python unit tests"
    python$PY $python_path/test/unit/runner.py -v2                                                || error "running python unit tests (serial)"
    progress "Python example: network_ring"
    python$PY $python_path/example/network_ring.py                                                || error "running python network_ring example"
    progress "Python example: single_cell_model"
    python$PY $python_path/example/single_cell_model.py                                           || error "running python single_cell_model example"
    progress "Python example: single_cell_recipe"
    python$PY $python_path/example/single_cell_recipe.py                                          || error "running python single_cell_recipe example"
    progress "Python example: single_cell_detailed"
    python$PY $python_path/example/single_cell_detailed.py $python_path/example/morph.swc         || error "running python single_cell_detailed example"
    progress "Python example: single_cell_detailed_recipe"
    python$PY $python_path/example/single_cell_detailed_recipe.py $python_path/example/morph.swc  || error "running python single_cell_detailed_recipe example"
    progress "Python example: single_cell_swc"
    python$PY $python_path/example/single_cell_swc.py  $base_path/test/unit/swc/pyramidal.swc     || error "running python single_cell_swc example"
    if [[ "${WITH_DISTRIBUTED}" = "mpi" ]]; then
        if [[ "$TRAVIS_OS_NAME" = "osx" ]]; then
            progress "Python distributed unit tests (MPI)"
            ${launch} python$PY $python_path/test/unit_distributed/runner.py -v2 || error "running python distributed unit tests (MPI)"
        else
            progress "Python distributed unit tests (MPI) -- skipping on Linux due to Travis config issues on Bionic."
        fi
    fi
fi

cd $base_path
