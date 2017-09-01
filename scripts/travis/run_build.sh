RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CLEAR='\033[0m'

error() {>&2 echo -e "${RED}ERROR${CLEAR}: $1"; exit 1;}
progress() { echo; echo -e "${YELLOW}STATUS${CLEAR}: $1"; echo;}

base_path=`pwd`
build_path=build-${BUILD_NAME}

#
# print build-specific and useful information
#
progress "build environment information"

compiler_version=`${CXX} -dumpversion`
cmake_version=`cmake --version | grep version | awk '{print $3}'`

echo "compiler   : ${compiler_version}"
echo "cmake      : ${cmake_version}"
echo "build path : ${build_path}"
echo "base path  : ${base_path}"

if [[ "${WITH_DISTRIBUTED}" = "mpi" ]]; then
    echo "mpi        : enabled"
    export OMPI_CC=${CC}
    export OMPI_CXX=${CXX}
    #CXX_FLAGS="-DCMAKE_CXX_FLAGS=-cxx=${CXX}"
    CC="mpicc"
    CXX="mpicxx"
    launch="mpiexec -n 4"
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

cmake_flags="-DNMC_WITH_ASSERTIONS=on -DNMC_THREADING_MODEL=${WITH_THREAD} -DNMC_DISTRIBUTED_MODEL=${WITH_DISTRIBUTED} ${CXX_FLAGS}"
echo "cmake flags: ${cmake_flags}"
cmake .. ${cmake_flags} || error "unable to configure cmake"

export NMC_NUM_THREADS=2

progress "Unit tests"
make test.exe -j4  || error "errors building unit tests"
./tests/test.exe || error "errors running unit tests"

progress "Global communication tests"
make global_communication.exe -j4          || error "errors building global communication tests"
${launch} ./tests/global_communication.exe || error "errors running global communication tests"

progress "Miniapp spike comparison test"
make miniapp.exe -j4                        || error "errors building miniapp"
${launch} ./miniapp/miniapp.exe -n 10 -t 10 || error "errors running miniapp"

cd $base_path
