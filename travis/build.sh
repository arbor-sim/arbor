$(CC) --version
$(CXX) --version
cmake --version

# make build path
mkdir -p build
cd build

# run cmake
cmake .. || exit 1
make || exit 1

# run tests
./tests/test.exe || exit 1
