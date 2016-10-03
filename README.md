# NestMC Prototype

This is the repository for the NestMC prototype code. Unfortunately we do not have thorough documentation of how-to guides.
Below are some guides for how to build the project and run the miniapp.
Contact us or submit a ticket if you have any questions or want help.

```bash
# clone repo
git clone git@github.com:eth-cscs/nestmc-proto.git
cd nestmc-proto/

# setup sub modules
git submodule init
git submodule update

# setup environment
# on a desktop system this is probably not required
# on a cluster this is usually required to make sure that an appropriate
# compiler is chosen.
module load gcc
module load cmake
export CC=`which gcc`
export CXX=`which g++`

# build main project (out-of-tree)
mkdir build
cd build
cmake ..
make -j

# test
cd tests
./test.exe
```

## MPI

Set the `WITH_MPI` option either via the ccmake interface, or via the command line as shown below.
To ensure that CMake detects MPI correctly, you should specify the MPI wrapper for the compiler by setting the `CXX` and `CC` environment variables.

```
export CXX=mpicxx
export CC=mpicc
cmake <path to CMakeLists.txt> -DWITH_MPI=ON

```

## TBB

Support for multi-threading requires Intel Threading Building Blocks (TBB).
When TBB is installed, it comes with some scripts that can be run to set up the user environment.
The scripts set the `TBB_ROOT` environment variable, which is used by the CMake configuration to find TBB.

```
source <path to TBB installation>/tbbvars.sh
cmake <path to CMakeLists.txt> -DWITH_TBB=ON
```

### TBB on Cray systems

To compile with TBB on Cray systems, load the intel module, which will automatically configure the environment.
The guide below shows how to use the version of TBB that is installed as part of the Intel compiler toolchain.
It is recommended that you install the most recent version of TBB yourself, and link against this, because older versions
of TBB don't work with recent versions of GCC.

```
# load the gnu environment for compiling the application
module load PrgEnv-gnu
# gcc 5.x does not work with the version of TBB installed on Cray
# requires at least version 4.4 of TBB
module swap gcc/4.9.3
# load the intel programming module
# on Cray systems this automatically sets `TBB_ROOT` environment variable
module load intel
module load cmake
export CXX=`which CC`
export CC=`which cc`

# multithreading only
cmake <path to CMakeLists.txt> -DWITH_TBB=ON -DSYSTEM_CRAY=ON

# multithreading and MPI
cmake <path to CMakeLists.txt> -DWITH_TBB=ON -DWITH_MPI=ON -DSYSTEM_CRAY=ON

```

# targetting KNL

## build modparser

The source to source compiler "modparser" that generates the C++/CUDA kernels for the ion channels and synapses is in a separate repository.
It is included in our project as a git submodule, and by default it will be built with the same compiler and flags that are used to build the miniapp and tests.

This can cause problems if we are cross compiling, e.g. for KNL, because the modparser compiler might not be runnable on the compilation node.
CMake will look for the source to source compiler executable, `modcc`, in the `PATH` environment variable, and will use the version if finds instead of building its own.

Modparser requires a C++11 compiler, and has been tested on GCC, Intel, and Clang compilers
  - if the default compiler on your is some ancient version of gcc you might need to load a module/set the CC and CXX environment variables.

```bash
git clone git@github.com:eth-cscs/modparser.git
cd modparser

# example of setting a C++11 compiler
export CXX=`which gcc-4.8`

cmake .
make -j

# set path and test that you can see modcc
export PATH=`pwd`/bin:$PATH
which modcc
```

## set up environment

- source the intel compilers
- source the TBB vars
- I have only tested with the latest stable version from online, not the version that comes installed sometimes with the Intel compilers.

## build miniapp

```bash
# clone the repo and set up the submodules
git clone https://github.com/eth-cscs/nestmc-proto.git
cd nestmc-proto
git submodule init
git submodule update

# make a path for out of source build
mkdir build_knl
cd build_knl

## build miniapp

# setup submodules
git submodule init
git submodule update

# make a path for out of source build
mkdir build_knl
cd build_knl

# run cmake with all the magic flags
export CC=`which icc`
export CXX=`which icpc`
cmake .. -DCMAKE_BUILD_TYPE=release -DWITH_TBB=ON -DWITH_PROFILING=ON -DVECTORIZE_TARGET=KNL -DUSE_OPTIMIZED_KERNELS=ON
make -j
```

The flags passed into cmake are described:
  - `-DCMAKE_BUILD_TYPE=release` : build in release mode with `-O3`.
  - `-WITH_TBB=ON` : use TBB for threading on multicore
  - `-DWITH_PROFILING=ON` : use internal profilers that print profiling report at end
  - `-DVECTORIZE_TARGET=KNL` : generate AVX512 instructions, alternatively you can use:
    - `AVX2` for Haswell & Broadwell
    - `AVX` for Sandy Bridge and Ivy Bridge
  - `-DUSE_OPTIMIZED_KERNELS=ON` : tell the source to source compiler to generate optimized kernels that use Intel extensions
    - without these vectorized code will not be generated.

## run tests

Run some unit tests
```bash
cd tests
./test.exe
cd ..
```

## run miniapp

The miniapp is the target for benchmarking.
First, we can run a small problem to check the build.
For the small test run, the parameters have the following meaning
  - `-n 1000` : 1000 cells
  - `-s 200` : 200 synapses per cell
  - `-t 20`  : simulated for 20ms
  - `-p 0`   : no file output of voltage traces

The number of cells is the number of discrete tasks that are distributed to the threads in each large time integration period.
The number of synapses per cell is the amount of computational work per cell/task.
Realistic cells have anywhere in the range of 1,000-10,000 synapses per cell.

```bash
cd miniapp

# a small run to check that everything works
./miniapp.exe -n 1000 -s 200 -t 20 -p 0

# a larger run for generating meaninful benchmarks
./miniapp.exe -n 2000 -s 2000 -t 100 -p 0
```

This generates the following profiler output (some reformatting to make the table work):

```
              ---------------------------------------
             |       small       |       large       |
-------------|-------------------|-------------------|
total        |  0.791     100.0  | 38.593     100.0  |
  stepping   |  0.738      93.3  | 36.978      95.8  |
    matrix   |  0.406      51.3  |  6.034      15.6  |
      solve  |  0.308      38.9  |  4.534      11.7  |
      setup  |  0.082      10.4  |  1.260       3.3  |
      other  |  0.016       2.0  |  0.240       0.6  |
    state    |  0.194      24.5  | 23.235      60.2  |
      expsyn |  0.158      20.0  | 22.679      58.8  |
      hh     |  0.014       1.7  |  0.215       0.6  |
      pas    |  0.003       0.4  |  0.053       0.1  |
      other  |  0.019       2.4  |  0.287       0.7  |
    current  |  0.107      13.5  |  7.106      18.4  |
      expsyn |  0.047       5.9  |  6.118      15.9  |
      pas    |  0.028       3.5  |  0.476       1.2  |
      hh     |  0.006       0.7  |  0.096       0.2  |
      other  |  0.026       3.3  |  0.415       1.1  |
    events   |  0.005       0.6  |  0.125       0.3  |
    sampling |  0.003       0.4  |  0.051       0.1  |
    other    |  0.024       3.0  |  0.428       1.1  |
  other      |  0.053       6.7  |  1.614       4.2  |
-----------------------------------------------------
```

