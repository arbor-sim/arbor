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
    export PYTHONPATH=$PYTHONPATH:${base_path}/${build_path}/lib
    python_path=$base_path/python
    echo "python src : ${python_path}"
    echo "PYTHONPATH : ${PYTHONPATH}"
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

cmake_flags="-DARB_WITH_ASSERTIONS=ON -DARB_WITH_MPI=${WITH_MPI} -DARB_WITH_PYTHON=${ARB_WITH_PYTHON} ${CXX_FLAGS}"
echo "cmake flags: ${cmake_flags}"
cmake .. ${cmake_flags} || error "unable to configure cmake"

export NMC_NUM_THREADS=2

progress "C++ unit tests"
make unit -j4                || error "building unit tests"
./bin/unit --gtest_color=no  || error "running unit tests"

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
    make pyarb -j4                                                           || error "building pyarb"
    progress "Python unit tests"
    python$PY $python_path/test/unit/runner.py -v2                           || error "running python unit tests (serial)"
    if [[ "${WITH_DISTRIBUTED}" = "mpi" ]]; then
        progress "Python distributed unit tests (MPI)"
        ${launch} python$PY $python_path/test/unit_distributed/runner.py -v2 || error "running python distributed unit tests (MPI)"
    fi
fi

cd $base_path
