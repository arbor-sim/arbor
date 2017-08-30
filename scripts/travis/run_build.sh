RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CLEAR='\033[0m'

error() {>&2 echo -e "${RED}ERROR${CLEAR}: $1";}
progress() { echo; echo -e "${YELLOW}STATUS${CLEAR}: $1"; echo;}

export CC=`gcc-6`
export CXX=`g++-6`

${CC} --version
${CXX} --version
cmake --version

base_path=`pwd`

# make build path
build_path=build-${BUILD_NAME}
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
make test.exe -j8
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
