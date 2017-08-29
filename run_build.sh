RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CLEAR='\033[0m'

error() {>&2 echo -e "${RED}ERROR${CLEAR}: $1";}
progress() { echo; echo -e "${YELLOW}STATUS${CLEAR}: $1"; echo;}

export CC=`which gcc-6`
export CXX=`which g++-6`

${CC} --version
${CXX} --version
cmake --version

# make build path
mkdir -p build
cd build

# run cmake
progress "configuring with cmake"
cmake ..
if [ $? -ne 0 ]; then
    error "unable to configure with cmake"
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
