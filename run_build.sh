$(CC) --version
$(CXX) --version
cmake --version

# make build path
mkdir -p build
cd build

# run cmake
echo "configuring with cmake"
cmake ..
if [ $? -ne 0 ]; then
    echo "Error: unable to configure with cmake"
    exit 2;
fi

# make the tests
echo "running make"
make test.exe -j8
if [ $? -ne 0 ]; then
    echo "Error: unable to build unit tests"
    exit 3;
fi

# run tests
echo "running tests"
./tests/test.exe
if [ $? -ne 0 ]; then
    echo "Error: some unit tests did not pass"
    exit 4;
fi
