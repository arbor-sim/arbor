RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CLEAR='\033[0m'

error() {>&2 echo -e "${RED}ERROR${CLEAR}: $1";}
progress() { echo; echo -e "${YELLOW}STATUS${CLEAR}: $1"; echo;}

export CC=`which gcc-6`
export CXX=`which g++-6`

base_path=`pwd`
build_path=build-${BUILD_NAME}

# print build-specific and useful information
progress "compiler versions"

compiler_version=`${CXX} -dumpversion`
cmake_version=`cmake --version | grep version | awk '{print $3}'`
echo "compiler   : ${compiler_version}"
echo "cmake      : ${cmake_version}"
echo "build path : ${build_path}"
echo "base path  : ${base_path}"

# make build path
mkdir -p $build_path
cd $build_path

# run cmake
progress "configuring with cmake"

cmake_flags="-DNMC_THREADING_MODEL=${WITH_THREAD}"
cmake .. ${cmake_flags}
if [ $? -ne 0 ]; then
    error "unable to configure with cmake ${cmake_flags}"
    exit 2;
fi

# make the tests
progress "running make"
make test.exe -j4
if [ $? -ne 0 ]; then
    error "unable to build unit tests"
    exit 3;
fi

# run tests
progress "running tests"
./tests/test.exe
if [ $? -ne 0 ]; then
    error "some unit tests did not pass"
    exit 4;
fi

cd $base_path
