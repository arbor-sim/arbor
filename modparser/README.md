# modparser

A source to source compiler for the NMODL domain specific language.

## Build and Test

### check out

first, check out the repository
```bash
# ssh
git clone git@github.com:eth-cscs/modparser.git
# https
git clone https://github.com/eth-cscs/modparser.git
```

There are no external dependencies. The project uses the following external projects, which are included in the repository (they don't have to be downloaded seperately)
* **Google Test** is used for the unit testing framework. The files for Google Test are part of the modparser repository, in ```tests/gtest```. BSD license.
* **TCLap (Templatized C++ Command Line Parser)** is used for command line parsing, is stored in `external/tclap`. MIT license.

### build

CMake and a C++11 compliant compiler are all that are required.

```bash
# you might want to specify the compiler, e.g. clang
export CC=`which clang`
export CXX=`which clang++`
cmake .
make all
```

```make all``` will build two targets: the ```bin/modcc``` executable for the compiler, and the unit tests ```tests/test_compiler```.

### test

To run the unit tests:
```bash
# you might want to update PATH to include modcc
export PATH=`pwd`/bin:$PATH
# run the unit tests
cd tests
./test_compiler
# check that there are no errors compiling the packaged example files
./test_compiler.sh
# If you have valgrind, you can check for memory leaks
# The valgrind tests have had some problems recently that appear to be caused by
# libc. The robustness of this testing needs to be improved
# (takes a couple of minutes)
./test_valgrind.sh
```

To test the compiler itself, first check that you can get help

```
./bin/modcc -help
```

There are some mod files in the ```tests/modfiles``` path, that can be used to generate some compiler output. Both an input file name and a target are required:

```
./bin/modcc tests/modfiles/KdShu2007.mod  -t gpu
./bin/modcc tests/modfiles/KdShu2007.mod  -t cpu
```

If no ouput file is specified, as above, the generated code is written to stdout. An output file can be specified using the ```-o``` flag

```
./bin/modcc tests/modfiles/KdShu2007.mod  -t gpu -o KdShu.h
```

### use

To use the compiler to generate the mechanism headers for the benchmark example @ github.com/eth-cscs/mod2c-perf, you will want to add the mod2c target to your PATH, e.g.
```
cd bin
export PATH=`pwd`:$PATH
```

