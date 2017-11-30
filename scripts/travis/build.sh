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

compiler_version=`${CXX} -dumpversion`
cmake_version=`cmake --version | grep version | awk '{print $3}'`

echo "compiler   : ${CXX} ${compiler_version}"
echo "cmake      : ${cmake_version}"
echo "build path : ${build_path}"
echo "base path  : ${base_path}"

if [[ "${WITH_DISTRIBUTED}" = "mpi" ]]; then
    echo "mpi        : on"
    export OMPI_CC=${CC}
    export OMPI_CXX=${CXX}
    CC="mpicc"
    CXX="mpicxx"
    launch="mpiexec -n 4"
else
    echo "mpi        : off"
    launch=""
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

cmake_flags="-DARB_WITH_ASSERTIONS=on -DARB_THREADING_MODEL=${WITH_THREAD} -DARB_DISTRIBUTED_MODEL=${WITH_DISTRIBUTED} ${CXX_FLAGS}"
echo "cmake flags: ${cmake_flags}"
cmake .. ${cmake_flags} || error "unable to configure cmake"

export NMC_NUM_THREADS=2

progress "Unit tests"
make test.exe -j4  || error "building unit tests"
./tests/test.exe --gtest_color=no || error "running unit tests"

progress "Global communication tests"
make global_communication.exe -j4          || error "building global communication tests"
${launch} ./tests/global_communication.exe || error "running global communication tests"

progress "Miniapp spike comparison test"
make miniapp.exe -j4                         || error "building miniapp"
${launch} ./miniapps/miniapp/miniapp.exe -n 20 -t 100 || error "running miniapp"

cd $base_path
